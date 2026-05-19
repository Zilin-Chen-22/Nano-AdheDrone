
#
# Note: 类名、方法名、参数名已经指定，请勿修改
#
#
# 
# @param param string字符串  
# @return string字符串
#
class Solution:
    def GetMinCalculateCount(self, sourceX, sourceY, targetX, targetY) :
        moveX = targetX
        moveY = targetY
        step = 0
        while True:
            if moveX == sourceX:
                if moveY == sourceY:
                    return step
                else:
                    return -1
            if moveX % 2 == 0 and (moveX / 2) >= sourceX and moveX != 2:
                moveX = moveX / 2
                moveY = moveY / 2
                step += 1
            else:
                moveX = moveX - 1
                moveY = moveY - 1
                step += 1
            print(moveX, " ", moveY)
            

    def check(self):
        print(self.GetMinCalculateCount(2,2,1,1))

sol = Solution()
sol.check()