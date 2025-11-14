package judge

import (
	"compress/gzip"
	"encoding/gob"
	"errors"
	"io"

	"github.com/mrhaoxx/go-mc/hpcworld"
)

type Dim struct {
	Chunks []hpcworld.Chunk
}

type TimeseriesChunk struct {
	Ticks      []Dim
	SampleRate int
}

type streamHeader struct {
	Count      int
	SampleRate int
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}

// WriteTo serializes the entire timeseries chunk into the provided writer using
// gob encoding wrapped in gzip compression. Dims are written sequentially to
// allow streaming readers to consume them incrementally.
func (t *TimeseriesChunk) WriteTo(w io.Writer) (int64, error) {
	if w == nil {
		return 0, errors.New("judge: nil writer")
	}

	sw, err := NewDimStreamWriter(w, t.SampleRate, len(t.Ticks))
	if err != nil {
		return 0, err
	}

	for _, dim := range t.Ticks {
		if err := sw.WriteDim(dim); err != nil {
			_ = sw.Close()
			return sw.BytesWritten(), err
		}
	}

	if err := sw.Close(); err != nil {
		return sw.BytesWritten(), err
	}

	return sw.BytesWritten(), nil
}

// BytesWritten exposes how many bytes passed through the counting writer.
func (c *countingWriter) BytesWritten() int64 {
	return c.n
}

// DimStreamWriter incrementally encodes Dim values into a gzip+gob stream.
type DimStreamWriter struct {
	cw         *countingWriter
	gz         *gzip.Writer
	enc        *gob.Encoder
	sampleRate int
	total      int
	written    int
	closed     bool
}

// NewDimStreamWriter builds a streaming writer over the provided writer. The
// total parameter controls the Count field in the stream header. Pass a
// negative value if the number of dims is unknown when the stream starts.
func NewDimStreamWriter(w io.Writer, sampleRate int, total int) (*DimStreamWriter, error) {
	if w == nil {
		return nil, errors.New("judge: nil writer")
	}

	cw := &countingWriter{w: w}
	gz := gzip.NewWriter(cw)
	enc := gob.NewEncoder(gz)

	if total < 0 {
		total = -1
	}

	hdr := streamHeader{
		Count:      total,
		SampleRate: sampleRate,
	}

	if err := enc.Encode(hdr); err != nil {
		_ = gz.Close()
		return nil, err
	}

	return &DimStreamWriter{
		cw:         cw,
		gz:         gz,
		enc:        enc,
		sampleRate: sampleRate,
		total:      total,
	}, nil
}

// WriteDim writes the next Dim into the stream. It enforces the Count declared
// in the header when non-negative.
func (s *DimStreamWriter) WriteDim(dim Dim) error {
	if s == nil || s.enc == nil {
		return errors.New("judge: stream writer not initialized")
	}
	if s.closed {
		return errors.New("judge: stream writer closed")
	}
	if s.total >= 0 && s.written >= s.total {
		return errors.New("judge: dim count exceeds header")
	}

	if err := s.enc.Encode(dim); err != nil {
		return err
	}

	s.written++
	return nil
}

// Close flushes and closes the underlying gzip stream.
func (s *DimStreamWriter) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.gz.Close()
}

// BytesWritten returns the number of bytes emitted so far.
func (s *DimStreamWriter) BytesWritten() int64 {
	if s == nil || s.cw == nil {
		return 0
	}
	return s.cw.BytesWritten()
}

// Written returns how many dims have been written to the stream.
func (s *DimStreamWriter) Written() int {
	if s == nil {
		return 0
	}
	return s.written
}

// DimStreamReader provides streaming access to compressed timeseries data at a
// Dim granularity. It decodes gob-serialized Dim values from a gzip reader.
type DimStreamReader struct {
	gz         *gzip.Reader
	dec        *gob.Decoder
	sampleRate int
	total      int
	read       int
	closed     bool
}

// NewDimStreamReader constructs a streaming reader over gzip+gob encoded
// TimeseriesChunk data. The caller must call Close when finished consuming dims.
func NewDimStreamReader(r io.Reader) (*DimStreamReader, error) {
	if r == nil {
		return nil, errors.New("judge: nil reader")
	}

	gz, err := gzip.NewReader(r)
	if err != nil {
		return nil, err
	}

	dec := gob.NewDecoder(gz)

	var hdr streamHeader
	if err := dec.Decode(&hdr); err != nil {
		_ = gz.Close()
		return nil, err
	}

	return &DimStreamReader{
		gz:         gz,
		dec:        dec,
		sampleRate: hdr.SampleRate,
		total:      hdr.Count,
	}, nil
}

// SampleRate returns the sample rate contained in the stream header.
func (s *DimStreamReader) SampleRate() int {
	return s.sampleRate
}

// Total returns the declared number of dims in the stream. A negative value
// indicates the producer did not specify a bound.
func (s *DimStreamReader) Total() int {
	return s.total
}

// Next decodes the next Dim. It returns ok=false when the stream has been fully
// consumed. Any decoding error is returned without wrapping when possible.
func (s *DimStreamReader) Next() (dim Dim, ok bool, err error) {
	if s == nil || s.dec == nil {
		return Dim{}, false, errors.New("judge: stream reader not initialized")
	}
	if s.closed {
		return Dim{}, false, errors.New("judge: stream reader closed")
	}
	if s.total >= 0 && s.read >= s.total {
		return Dim{}, false, nil
	}

	if err := s.dec.Decode(&dim); err != nil {
		if errors.Is(err, io.EOF) {
			if s.total >= 0 && s.read < s.total {
				return Dim{}, false, io.ErrUnexpectedEOF
			}
			return Dim{}, false, nil
		}
		return Dim{}, false, err
	}

	s.read++
	return dim, true, nil
}

// Close releases underlying resources associated with the reader.
func (s *DimStreamReader) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.gz.Close()
}

// ReadTimeseriesChunk fully materializes the compressed stream into a
// TimeseriesChunk instance.
func ReadTimeseriesChunk(r io.Reader) (*TimeseriesChunk, error) {
	sr, err := NewDimStreamReader(r)
	if err != nil {
		return nil, err
	}
	defer sr.Close()

	total := sr.Total()
	if total < 0 {
		total = 0
	}

	chunk := &TimeseriesChunk{
		SampleRate: sr.SampleRate(),
		Ticks:      make([]Dim, 0, total),
	}

	for {
		dim, ok, err := sr.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		chunk.Ticks = append(chunk.Ticks, dim)
	}

	return chunk, nil
}
