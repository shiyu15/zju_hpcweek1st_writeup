# 解答
好像之前在zjuctf上做过类似的题目，但是时间太久记不清了。

从网上抄几个quine来测试一下
```bash
_ ()
{
    echo "$(local -f _);_"
};_
```

```bash
_ ()
{
    function __ ()
    {
        true
    };
    ${1} &> /dev/null;
    echo "$(declare -f _);_ ${@}"
};_ __
```
不过上面这两个都无法过测试。

最后在下面这个仓库找到了可以过测试的quine
https://github.com/makenowjust/quine
```bash
b=\' c=\\ a='echo -n b=$c$b c=$c$c a=$b$a$b\;; echo $a';echo -n b=$c$b c=$c$c a=$b$a$b\;; echo $a

```

听同学说单个空格也可以过，可能是题意中的"最简单的答案可能会比想象中的要简短很多"