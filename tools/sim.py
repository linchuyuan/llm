import pdb
import sys

sys.setrecursionlimit(6000)
LOG = True
LOSS_SCALE = 3
PERCENTAGE_UP = 0.48
PERCENTAGE_DOWN = (1 - PERCENTAGE_UP)

def log(string):
    if LOG:
        print(string)

def dfs(mep, p2, p1, put_setpoint, call_setpoint, cur, max_step=10,
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
        is_put = False
        premium = 1
        if p1 >= p2:
            is_put = True
            premium = 1
        if step >= max_step:
            if (is_put and cur >= p1) or (not is_put and cur <= p1):
                log("winning premium case cur %s, step %s, p1 %s, p2 %s, is_put %s" %( cur,
                    step, p1, p2, is_put))
                exp = premium
                total = total_case + 1
                win = win_case
                if exp > 0:
                    win += 1
            elif (is_put and cur >= put_setpoint) or (not is_put and cur <= call_setpoint):
                log("loss case found 1, cur %s, step %s" % (cur, step))
                exp = -premium * LOSS_SCALE
                total = total_case + 1
                win = win_case
                if exp > 0:
                    win += 1
            else:
                log("winning case cur %s, step %s, p1 %s, p2 %s, is_put %s" %( cur, 
                    step, p1, p2, is_put))
                exp = abs(cur - p2) - (premium * LOSS_SCALE)
                total = total_case + 1
                win = win_case
                if exp > 0:
                    win +=  1
        elif cur == p1:
            log("remove p1 %s, cur %s, step %s" % (p1, cur, step))
            # no more p1
            if is_put:
                expected_u, total_u, win_u = dfs(mem, p2, 0xffff, put_setpoint, 0xffff, cur+1,
                    max_step, step+1, expected, total_case, win_case)
                expected_d, total_d, win_d = dfs(mem, p2, 0xffff, put_setpoint, 0xffff, cur-1,
                    max_step, step+1, expected, total_case, win_case)
            else:
                expected_u, total_u, win_u = dfs(mem, p2, 0, 0, call_setpoint, cur+1,
                    max_step, step+1, expected, total_case, win_case)
                expected_d, total_d, win_d = dfs(mem, p2, 0, 0, call_setpoint, cur-1,
                    max_step, step+1, expected, total_case, win_case)
            exp = PERCENTAGE_UP * expected_u + PERCENTAGE_DOWN * expected_d
            total = total_u + total_d
            win = win_u + win_d
        else:
            expected_u, total_u, win_u = dfs(mem, p2, p1, put_setpoint, call_setpoint, cur+1,
                max_step, step+1, expected, total_case, win_case)
            expected_d, total_d, win_d = dfs(mem, p2, p1, put_setpoint, call_setpoint, cur-1,
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


mem = dict()
# is put spread no call_setpoint
exp, total, win = dfs(mem, 498, 499, 497, 0xffff, 500, max_step=50, step=0)

print("expected value %.20f, winrate %.2f%%" %(exp,  (win/total)*100))

