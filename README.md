<h1 align="center">刷题计划</h1>

# 2024/5/7
## 27.移除元素
### 题目描述：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给你一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，并返回移除后数组的新长度。不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.使用双指针的思路,一个"慢"指针 i,表示当前已处理好的数组的长度。一个"快"指针 j,用于遍历整个数组。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.遍历数组,对于每个元素: 如果当前元素不等于 val,则将其复制到i指针的位置,并将i向右移动一步。如果当前元素等于 val,则跳过该元素,i指针不动。遍历结束后,i的值就是新数组的长度。
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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.我们可以利用数组A已经按非递减顺序排序的性质来思考:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.数组平方的最大值就在数组的两端,不是最左边就是最右边,不可能是中间。可以考虑双指针法,左右指针分别从数组两端向中间移动,每次将平方后数值较大的元素放入结果数组。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.首先确定数组的长度 n,并创建左右指针 left 和 right,分别指向数组的左右两端。创建一个长度为 n 的新数组 result 用于存储结果,同时创建一个指针 index 指向 result 数组的终止位置 n-1。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.开始循环,当 left 小于等于 right 时:比较 nums[left] 和 nums[right] 的绝对值大小(这里用了一个辅助函数 abs)如果 nums[right] 的绝对值更大,将 nums[right] 的平方值放入 result[index]将 right 指针向左移动。    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.否则,nums[left] 的绝对值更大或相等:将 nums[left] 的平方值放入 result[index],将 left 指针向右移动,将 index 指针向左移动,循环结束后,返回 result 数组即为所求结果。
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
# 2024/5/13
## 203. 移除链表元素
### 题目描述：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回新的头节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入：head = [1,2,6,3,4,5,6], val = 6  输出：[1,2,3,4,5]  
### 解题思路:
这个题的主要思路是遍历链表,找到值等于给定值的节点,然后删除这个节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.首先,我们需要创建一个虚拟头节点(dummy node),指向链表的头节点。这样做的目的是为了处理头节点可能被删除的情况。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.然后,我们使用两个指针 prev 和 curr,prev 指向当前节点的前一个节点,curr 指向当前节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.遍历链表,如果当前节点的值等于给定值,那么我们将 prev 的 next 指针指向 curr 的下一个节点,这样就删除了当前节点。如果当前节点的值不等于给定值,我们将 prev 移动到当前节点,curr 移动到下一个节点。  
### 代码实现
```
type ListNode struct {
    Val  int
    Next *ListNode
}

func removeElements(head *ListNode, val int) *ListNode {
    dummy := &ListNode{Next: head}
    prev, curr := dummy, head

    for curr != nil {
        if curr.Val == val {
            prev.Next = curr.Next
        } else {
            prev = curr
        }
        curr = curr.Next
    }

    return dummy.Next
}
```
## 707. 设计链表
### 题目描述：
使用单链表或者双链表，设计并实现自己的链表。  
实现 MyLinkedList 类：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.MyLinkedList() 初始化 MyLinkedList 对象。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.int get(int index) 获取链表中下标为 index 的节点的值。如果下标无效，则返回 -1 。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.void addAtHead(int val) 将一个值为 val 的节点插入到链表中第一个元素之前。在插入完成后，新节点会成为链表的第一个节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.void addAtTail(int val) 将一个值为 val 的节点追加到链表中作为链表的最后一个元素。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.void addAtIndex(int index, int val) 将一个值为 val 的节点插入到链表中下标为 index 的节点之前。如果 index 等于链表的长度，那么该节点会被追加到链表的末尾。如果 index 比长度更大，该节点将 不会插入 到链表中。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.void deleteAtIndex(int index) 如果下标有效，则删除链表中下标为 index 的节点。  
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.首先,我们定义一个 Node 结构体,表示链表中的节点。每个节点包含一个 val 属性,表示节点的值,以及一个 next 属性,表示指向下一个节点的指针。    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.然后,我们定义一个 MyLinkedList 结构体,表示链表本身。它包含一个虚拟头节点 dummy,以及一个表示链表长度的 size 属性。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.get(index):我们从虚拟头节点开始,向后遍历 index 次,如果到达链表末尾,说明索引无效,返回 -1,否则返回对应节点的值。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.addAtHead(val):我们创建一个新节点,将其 next 指针指向虚拟头节点的下一个节点,然后将虚拟头节点的 next 指针指向新节点。最后,将链表长度加 1。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.addAtTail(val):我们创建一个新节点,从虚拟头节点开始,遍历到链表末尾,将末尾节点的 next 指针指向新节点。最后,将链表长度加 1。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.addAtIndex(index, val):如果 index 等于链表长度,则调用 addAtTail 方法。如果 index 小于0,则调用 addAtHead 方法。否则,我们从虚拟头节点开始,向后遍历 index 次,将新节点插入到对应位置。最后,将链表长度加 1。    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.deleteAtIndex(index):如果索引无效,直接返回。否则,我们从虚拟头节点开始,向后遍历 index 次,将对应节点的前一个节点的 next 指针指向对应节点的下一个节点。最后,将链表长度减 1。  
### 代码实现
```
type Node struct {
    Val  int
    Next *Node
}

type MyLinkedList struct {
    dummy *Node
    size  int
}

func Constructor() MyLinkedList {
    return MyLinkedList{&Node{}, 0}
}

func (this *MyLinkedList) Get(index int) int {
    if index < 0 || index >= this.size {
        return -1
    }
    curr := this.dummy.Next
    for i := 0; i < index; i++ {
        curr = curr.Next
    }
    return curr.Val
}

func (this *MyLinkedList) AddAtHead(val int) {
    node := &Node{Val: val}
    node.Next = this.dummy.Next
    this.dummy.Next = node
    this.size++
}

func (this *MyLinkedList) AddAtTail(val int) {
    curr := this.dummy
    for curr.Next != nil {
        curr = curr.Next
    }
    curr.Next = &Node{Val: val}
    this.size++
}

func (this *MyLinkedList) AddAtIndex(index int, val int) {
    if index > this.size {
        return
    }
    if index < 0 {
        index = 0
    }
    curr := this.dummy
    for i := 0; i < index; i++ {
        curr = curr.Next
    }
    node := &Node{Val: val}
    node.Next = curr.Next
    curr.Next = node
    this.size++
}

func (this *MyLinkedList) DeleteAtIndex(index int) {
    if index < 0 || index >= this.size {
        return
    }
    curr := this.dummy
    for i := 0; i < index; i++ {
        curr = curr.Next
    }
    curr.Next = curr.Next.Next
    this.size--
}
```
##  206.反转链表 
### 题目描述：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。输入：head = [1,2,3,4,5]&emsp;输出：[5,4,3,2,1]
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.初始化三个指针:prev、curr和next,分别表示当前节点的前一个节点、当前节点和当前节点的下一个节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.将prev初始化为null,curr初始化为头节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.遍历链表,对于每个节点:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.将next指向curr的下一个节点,保存当前节点的下一个节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.将curr的next指针指向prev,反转当前节点的指向。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.将prev移动到curr,curr移动到next,继续下一次迭代。当curr为null时,prev就是新的头节点。  
### 代码实现
```
type ListNode struct {
    Val  int
    Next *ListNode
}

// 迭代法
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    curr := head

    for curr != nil {
        next := curr.Next
        curr.Next = prev
        prev = curr
        curr = next
    }

    return prev
}
```
# 2024/5/14
##  24. 两两交换链表中的节点
### 题目描述：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。输入：head = [1,2,3,4]&emsp;输出：[2,1,4,3]
### 解题思路:

