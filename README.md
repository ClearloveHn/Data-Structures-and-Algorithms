<h1 align="center">刷题计划</h1>

# 2024/5/7
## 27.移除元素
### 题目描述：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给你一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，并返回移除后数组的新长度。不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用双指针的思路,一个"慢"指针 i,表示当前已处理好的数组的长度。一个"快"指针 j,用于遍历整个数组。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;遍历数组,对于每个元素: 如果当前元素不等于 val,则将其复制到i指针的位置,并将i向右移动一步。如果当前元素等于 val,则跳过该元素,i指针不动。遍历结束后,i的值就是新数组的长度。
### 代码实现
```
func removeElement(nums []int, val int) int {
   i := 0 // 定义i指针

   // j指针遍历数组
   for j := 0;j < len(nums);j++ {
    if nums[j] != val {
        nums[i] = nums[j]
        i++
    }
   }
   return i
}
```
## 977.有序数组的平方
### 题目描述:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给你一个按非递减顺序排序的整数数组 nums，返回每个数字的平方组成的新数组，要求也按非递减顺序排序。
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们可以利用数组A已经按非递减顺序排序的性质来思考:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;数组平方的最大值就在数组的两端,不是最左边就是最右边,不可能是中间。可以考虑双指针法,左右指针分别从数组两端向中间移动,每次将平方后数值较大的元素放入结果数组。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先确定数组的长度 n,并创建左右指针 left 和 right,分别指向数组的左右两端。创建一个长度为 n 的新数组 result 用于存储结果,同时创建一个指针 index 指向 result 数组的终止位置 n-1。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;开始循环,当 left 小于等于 right 时:比较 nums[left] 和 nums[right] 的绝对值大小(这里用了一个辅助函数 abs)如果 nums[right] 的绝对值更大,将 nums[right] 的平方值放入 result[index]将 right 指针向左移动。    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;否则,nums[left] 的绝对值更大或相等:将 nums[left] 的平方值放入 result[index],将 left 指针向右移动,将 index 指针向左移动,循环结束后,返回 result 数组即为所求结果。
### 代码实现
```
func sortedSquares(nums []int) []int {
     n:=len(nums)
     left,right := 0,n - 1
     result := make([]int,n)
     index := n-1
     for left <= right {
        if abs(nums[left]) < abs(nums[right]) {
            result[index] = nums[right] * nums[right]
            right --
        } else {
            result[index] = nums[left] * nums[left]
            left ++
        }
        index --
     }
     return result

}

func abs(x int) int {
    if x < 0 {
        return -x
    } 
    return x
}
```
