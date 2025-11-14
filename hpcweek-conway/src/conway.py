# return: grid, (dy, dx)
# (dy, dx) indicates the movement of upper left corner, only used for visualization
def Next_Generation_Ref(grid):
    if not grid or not grid[0]:
        return [], (0, 0)

    height, width = len(grid), len(grid[0])
    padded_grid = [[0] * (width + 2) for _ in range(height + 2)]
    for y in range(height):
        for x in range(width):
            padded_grid[y + 1][x + 1] = grid[y][x]

    next_padded_grid = [[0] * (width + 2) for _ in range(height + 2)]
    
    for y in range(height + 2):
        for x in range(width + 2):
            live_neighbors = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    if y + i >= 0 and y + i < height + 2 and x + j >= 0 and x + j < width + 2:
                        live_neighbors += padded_grid[y + i][x + j]

            next_padded_grid[y][x] = live_neighbors == 3 or (live_neighbors == 2 and padded_grid[y][x] == 1)

    min_y, max_y = -1, -1
    min_x, max_x = -1, -1

    for y in range(height + 2):
        for x in range(width + 2):
            if next_padded_grid[y][x] == 1:
                if min_y == -1 or y < min_y: 
                    min_y = y
                if max_y == -1 or y > max_y: 
                    max_y = y
                if min_x == -1 or x < min_x: 
                    min_x = x
                if max_x == -1 or x > max_x: 
                    max_x = x
    
    if min_y == -1:
        return [], (0, 0)

    new_grid = []
    for y in range(min_y, max_y + 1):
        new_grid.append(list(next_padded_grid[y][min_x : max_x + 1]))

    return new_grid, (min_y - 1, min_x - 1)

def Expand_Ref(grid, iter):
    for _ in range(iter):
        # for row in grid:
        #     print(row)
        # print("")
        prev_grid = [row[:] for row in grid]
        grid, _ = Next_Generation_Ref(grid)
        if grid == prev_grid:
            break
    return grid

# ========= modifiable ==========

# import NG
# import numpy as np


# def Next_Generation(grid):
#     if not grid or not grid[0]:
#         return [], (0, 0)

#     np_grid = np.array(grid, dtype=np.uint8)


#     padded = np.pad(np_grid, pad_width=1, mode='constant', constant_values=0)

#     neighbor_count = np.zeros_like(padded, dtype=np.uint8)
#     neighbor_count[1:, 1:] += padded[:-1, :-1]
#     neighbor_count[1:, :] += padded[:-1, :]
#     neighbor_count[1:, :-1] += padded[:-1, 1:]
#     neighbor_count[:, 1:] += padded[:, :-1]
#     neighbor_count[:, :-1] += padded[:, 1:]
#     neighbor_count[:-1, 1:] += padded[1:, :-1]
#     neighbor_count[:-1, :] += padded[1:, :]
#     neighbor_count[:-1, :-1] += padded[1:, 1:]

#     next_generation = ((neighbor_count == 3) | ((neighbor_count == 2) & (padded == 1))).astype(np.uint8)

#     coords = np.argwhere(next_generation == 1)
#     if coords.size == 0:
#         return [], (0, 0)

#     min_y, min_x = coords.min(axis=0)
#     max_y, max_x = coords.max(axis=0)

#     new_grid_np = next_generation[min_y:max_y + 1, min_x:max_x + 1]
#     new_grid = new_grid_np.tolist()

#     return new_grid, (min_y - 1, min_x - 1)



# def Expand(grid, iter):
#     for _ in range(iter):
#         # for row in grid:
#         #     print(row)
#         # print("")
#         prev_grid = [row[:] for row in grid]
#         grid, _ = Next_Generation(grid)
#         grid = grid.tolist()
#         if grid == prev_grid:
#             break
#     return grid


# import numpy as np
# from scipy.ndimage import uniform_filter  # 比 convolve2d 更快的 3x3 求和

# # ------------------- 单步：多态版（ndarray进 -> ndarray出；list进 -> list出） -------------------
# def Next_Generation(grid):
#     """
#     - 输入是 ndarray  -> 返回 (ndarray, (off_y, off_x))
#     - 输入是 list[list[int]] -> 返回 (list[list[int]], (off_y, off_x))
#     除了入/出转换，整个流程都在 NumPy/SciPy 上完成。
#     语义与原实现一致：先 pad(1)，在 padded 平面上计算与裁剪，offset=(min_y-1, min_x-1)。
#     """
#     is_list_input = not isinstance(grid, np.ndarray)
#     np_grid = np.asarray(grid, dtype=np.uint8, order='C') if is_list_input else grid
#     if np_grid.size == 0:
#         return ([], (0, 0)) if is_list_input else (np.zeros((0, 0), dtype=np.uint8), (0, 0))

#     # 1) pad 一圈 0（与原版保持一致）
#     padded = np.pad(np_grid, pad_width=1, mode='constant', constant_values=0)

#     # 2) 3×3 求和：用 uniform_filter（均值×9 即求和），再减掉中心（避免把自身计入邻居）
#     #    uniform_filter 返回浮点，乘以 9 后仍为浮点；最终统一转回 uint8 即可。
#     nb_sum = uniform_filter(padded.astype(np.float32, copy=False), size=3, mode='constant', cval=0.0)
#     nb_sum *= 9.0
#     # 减中心（padded 为 0/1），得到 8 邻居计数
#     neighbor_count = (nb_sum - padded).astype(np.uint8, copy=False)

#     # 3) 生命游戏规则（仍在 padded 空间）
#     #    注意：把逻辑运算放在 u8 -> bool 上，最后再 cast 回 u8，避免生成大临时
#     alive = (padded == 1)
#     next_generation = np.where((neighbor_count == 3) | ((neighbor_count == 2) & alive), 1, 0).astype(np.uint8, copy=False)

#     # 4) 裁剪最小包围盒，并返回 offset（与原版一致）
#     if not next_generation.any():
#         return ([], (0, 0)) if is_list_input else (np.zeros((0, 0), dtype=np.uint8), (0, 0))

#     ys, xs = np.nonzero(next_generation)   # 比 argwhere 更省内存
#     min_y, max_y = ys.min(), ys.max()
#     min_x, max_x = xs.min(), xs.max()
#     trimmed = next_generation[min_y:max_y+1, min_x:max_x+1]
#     offset = (min_y - 1, min_x - 1)

#     return (trimmed.tolist(), offset) if is_list_input else (trimmed, offset)


# # ------------------- 多步：入口/出口 list[list]，中间全 ndarray -------------------
# def Expand(grid, iter):
#     """
#     入口/出口是 list[list[int]]；内部每一步都用 ndarray。
#     与原版一致：稳态提前退出。
#     """
#     cur = np.asarray(grid, dtype=np.uint8, order='C') if (grid and grid[0]) else np.zeros((0, 0), dtype=np.uint8)

#     for _ in range(iter):
#         nxt, _ = Next_Generation(cur)  # 传 ndarray -> 返回 ndarray（不会触发 list 转换）
#         # 稳态提前退出（无需 prev=cur.copy()）
#         if nxt.shape == cur.shape and np.array_equal(nxt, cur):
#             cur = nxt
#             break
#         cur = nxt

#     return cur.tolist()

import NG 
def Expand(grid, iter):
    # Implement your own version to calculate the final grid
    return NG.Expand_Cpp(grid, iter)