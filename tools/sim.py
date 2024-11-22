import pdb
import sys

sys.setrecursionlimit(6000)
LOG = False
LOSS_SCALE = 8.3
PERCENTAGE_UP = 0.5
PERCENTAGE_DOWN = (1 - PERCENTAGE_UP)

def log(string):
    if LOG:
        print(string)

def dfs(mep, p2, p1, cur, max_step=10,
        step=0, expected=0., total_case=0, win_case=0):
    exp = None
    total = None
    win = None
    if cur in mem and step in mem[cur] and p1 in mem[cur][step]:
        exp, total, win = mem[cur][step][p1]
        log("cache hit, cur %s, p1 %s, step %s, exp %s, total %s, win %s" %(cur, p1, step, exp, total, win))
        return mem[cur][step][p1]
    else:
        log("current %s, step %s, p1 %s" % (cur, step, p1))
        is_put = (p1 >= p2)
        premium = 1
        if step >= max_step:
            if (is_put and cur >= p1) or (not is_put and cur <= p1):
                log("winning premium case cur %s, step %s, p1 %s, p2 %s, is_put %s" %( cur,
                    step, p1, p2, is_put))
                exp = premium
            elif (is_put and cur >= p2) or (not is_put and cur <= p1):
                log("loss case found 1, cur %s, step %s" % (cur, step))
                exp = -premium * LOSS_SCALE
            elif(is_put and p1 != 0xffff) or (not is_put and p1 != 0):
                # not stand alone options
                log("loss case found 1, cur %s, step %s" % (cur, step))
                exp = -premium * LOSS_SCALE
            else:
                # stand alone options
                raise
                log("winning case cur %s, step %s, p1 %s, p2 %s, is_put %s" %( cur, 
                    step, p1, p2, is_put))
                exp = abs(cur - p2) - (premium * LOSS_SCALE)
            total = total_case + 1
            win = win_case + (1 if exp > 0 else 0)
        elif cur == p1 and False:
            log("remove p2 %s, cur %s, step %s" % (p2, cur, step))
            # no more p1 and effectively move p2 to a lower number for put spread or higher for call spread
            if is_put:
                expected_u, total_u, win_u = dfs(mem, p2-(abs(p1-p2)), 0xffff, cur+.02,
                    max_step, step+1, expected, total_case, win_case)
                expected_d, total_d, win_d = dfs(mem, p2-(abs(p1-p2)), 0xffff, cur-.02,
                    max_step, step+1, expected, total_case, win_case)
            else:
                expected_u, total_u, win_u = dfs(mem, p2+(abs(p1-p2)), 0, cur+.02,
                    max_step, step+1, expected, total_case, win_case)
                expected_d, total_d, win_d = dfs(mem, p2+(abs(p1-p2)), 0, cur-.02,
                    max_step, step+1, expected, total_case, win_case)
            exp = PERCENTAGE_UP * expected_u + PERCENTAGE_DOWN * expected_d
            total = total_u + total_d
            win = win_u + win_d
        else:
            expected_u, total_u, win_u = dfs(mem, p2, p1, cur+.02,
                max_step, step+1, expected, total_case, win_case)
            expected_d, total_d, win_d = dfs(mem, p2, p1, cur-.02,
                max_step, step+1, expected, total_case, win_case)
            exp = PERCENTAGE_UP * expected_u + PERCENTAGE_DOWN * expected_d
            total = total_u + total_d
            win = win_u + win_d
    if cur not in mem:
        mem[cur] = dict()
    if step not in mem[cur]:
        mem[cur][step] = dict()
    exp = expected + exp
    log("set cache cur %s, p1 %s, step %s, exp %s, total %s, win %s" %(cur, p1, step, exp, total, win))
    mem[cur][step][p1] = (exp, total, win)
    return mem[cur][step][p1]

for p1 in range(500, 570):
    for spread in range(5, 6):
        p2 = p1 - spread
        mem = dict()
        exp, total, win = dfs(mem, p2, p1, 560, max_step=2000, step=0)
        print("expected value %.20f, winrate %.2f%%, p2 %s, p1 %s" %(exp,  (win/total)*100, p2, p1))


