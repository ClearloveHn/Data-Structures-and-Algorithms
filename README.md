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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.首先，我们需要一个哑节点（dummy node）作为头节点的前一个节点，以便处理头节点的交换。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.然后,我们使用一个指针 curr 初始指向哑节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.当 curr.Next 和 curr.Next.Next 都不为空时,进行以下步骤  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.保存 curr.Next 为 first,保存 curr.Next.Next 为 second,即需要交换的两个节点。。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2交换 first 和 second 的位置:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.1将 first.Next 指向 second.Next,即将第一个节点的下一个指针指向第二个节点的下一个节点。 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.2将 second.Next 指向 first,即将第二个节点的下一个指针指向第一个节点  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.3将 curr.Next 指向 second,即将当前节点的下一个指针指向交换后的第二个节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.更新指针：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.1将 curr 移动到 first,即移动到已交换的第一个节点上。  
### 代码实现
```
type ListNode struct {
    Val int
    Next *ListNode
}

func swapPairs(head *ListNode) *ListNode {
    dummy := &ListNode{Next: head}
    curr := dummy

    for curr.Next != nil && curr.Next.Next != nil {
        first := curr.Next
        second := curr.Next.Next

        first.Next = second.Next
        second.Next = first
        curr.Next = second

        curr = first
    }

    return dummy.Next
}
```
##  19.删除链表的倒数第N个节点  
### 题目描述：
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。&emsp;输入：head = [1,2,3,4,5], n = 2 &emsp;输出：[1,2,3,5]
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.使用两个指针 fast 和 slow，初始时都指向链表的头节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.先让 fast 指针向前移动 N 个节点，如果在移动过程中 fast 指针为空，说明链表长度不足 N，直接返回头节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.然后让 fast 和 slow 指针同时向前移动，直到 fast 指针到达链表的末尾。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.此时 slow 指针指向的节点就是倒数第 N 个节点的前一个节点。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.将 slow 指针的 Next 指针指向 slow.Next.Next，即删除倒数第 N 个节点。
### 代码实现
```
type ListNode struct {
    Val int
    Next *ListNode
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
    // 创建一个哑节点，指向链表的头节点
    dummy := &ListNode{Next: head}
    fast, slow := dummy, dummy

    // 先让 fast 指针向前移动 N 个节点
    for i := 0; i < n; i++ {
        fast = fast.Next
    }

    // 如果 fast 指针为空，说明链表长度不足 N
    if fast == nil {
        return head
    }

    // 让 fast 和 slow 指针同时向前移动，直到 fast 指针到达链表的末尾
    for fast.Next != nil {
        fast = fast.Next
        slow = slow.Next
    }

    // 删除倒数第 N 个节点
    slow.Next = slow.Next.Next

    return dummy.Next
}
```
##  142.环形链表 II
### 题目描述：
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.使用快慢指针法,快指针每次移动两步,慢指针每次移动一步。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.如果链表中存在环,快慢指针一定会在环内相遇。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.当快慢指针相遇时,将其中一个指针移动到链表头部,另一个指针保持在相遇位置。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.两个指针同时以相同的速度移动,它们会在环的入口处相遇。  
### 代码实现
```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func detectCycle(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return nil
    }
    
    // 快慢指针
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        // 快慢指针相遇
        if slow == fast {
            // 将其中一个指针移动到链表头部
            slow = head
            
            // 两个指针同时以相同的速度移动
            for slow != fast {
                slow = slow.Next
                fast = fast.Next
            }
            
            // 返回环的入口节点
            return slow
        }
    }
    
    // 链表中不存在环
    return nil
}
```
##  242.有效的字母异位词
### 题目描述：
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.首先，我们可以对字符串 s 和 t 进行长度判断，如果长度不相等，则直接返回 false，因为长度不同的字符串不可能是字母异位词。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.如果长度相等，我们可以使用哈希表（映射）来统计字符串 s 中每个字符出现的次数。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.然后，我们遍历字符串 t 中的每个字符，在哈希表中查找该字符并将其计数减一。如果发现某个字符在哈希表中不存在或计数已经为零，则说明 t 包含了 s 中没有的额外字符，返回 false。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.如果遍历完字符串 t 后，哈希表中所有字符的计数都为零，则说明 s 和 t 是字母异位词，返回 true。  
### 代码实现
```
func isAnagram(s string, t string) bool {
    // 如果两个字符串长度不相等，直接返回 false
    if len(s) != len(t) {
        return false
    }
    
    // 创建一个哈希表来统计字符出现的次数
    charCount := make(map[rune]int)
    
    // 遍历字符串 s，统计每个字符出现的次数
    for _, char := range s {
        charCount[char]++
    }
    
    // 遍历字符串 t，在哈希表中查找并减少字符的计数
    for _, char := range t {
        count, exists := charCount[char]
        if !exists || count == 0 {
            return false
        }
        charCount[char]--
    }
    
    // 如果遍历完 t 后，哈希表中所有字符的计数都为零，则返回 true
    return true
}
```
##  349.两个数组的交集
### 题目描述：
给定两个数组 nums1 和 nums2 ，返回它们的交集。输出结果中的每个元素一定是唯一 的。我们可以不考虑输出结果的顺序 。
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.创建一个哈希表（map）来存储第一个数组 nums1 中的所有元素，以便快速查找。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.创建一个结果数组 result，用于存储两个数组的交集元素。     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.遍历第二个数组 nums2 中的每个元素：如果当前元素在哈希表中存在，则将其添加到结果数组中，并从哈希表中删除该元素，确保交集中的元素唯一。  
### 代码实现
```
func intersection(nums1 []int, nums2 []int) []int {
    // 创建哈希表存储 nums1 中的元素
    set := make(map[int]bool)
    for _, num := range nums1 {
        set[num] = true
    }
    
    // 创建结果数组
    var result []int
    
    // 遍历 nums2，检查元素是否在哈希表中存在
    for _, num := range nums2 {
        if set[num] {
            result = append(result, num)
            delete(set, num) // 从哈希表中删除该元素，确保交集中的元素唯一
        }
    }
    
    return result
}
```
## 202.快乐数 
### 题目描述：
编写一个算法来判断一个数 n 是不是快乐数。「快乐数」 定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果这个过程结果为 1，那么这个数就是快乐数。  
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.将数字转换成每个位上的数字的平方和。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.判断平方和是否为 1，若为 1，则是快乐数。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.若不是 1，则重复步骤 1，直到得到 1 或者进入无限循环。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.为了判断是否进入无限循环，可以使用快慢指针法，若快慢指针相遇，则说明进入了无限循环。  
### 代码实现
```
func isHappy(n int) bool {
	slow, fast := n, step(n)
	// 判断平方和是否为 1
	for fast != 1 && slow != fast {
		slow = step(slow)
		fast = step(step(fast))
	}
	return fast == 1
}

// 数字转换成每个位上的数字的平方和。
func step(n int) int {
	sum := 0
	for n > 0 {
		digit := n % 10 // 取n的个位数
		sum += digit * digit
		n /= 10 // 去掉最后一位数
	}
	return sum
}
```
##  1.两数之和
### 题目描述：
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
### 解题思路:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.使用哈希表(map)来存储数组中每个元素的值和它的下标。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.遍历数组,对于每个元素 nums[i]:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1: 计算 complement = target - nums[i],在哈希表中查找是否存在 complement。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2: 如果存在,并且它的下标不等于当前元素的下标 i,则找到了目标和,返回两个元素的下标。    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3:如果不存在,将当前元素的值和下标存入哈希表。   
### 代码实现
```
func twoSum(nums []int, target int) []int {
    // 创建哈希表
    numMap := make(map[int]int)
    
    // 遍历数组
    for i, num := range nums {
        // 计算 complement
        complement := target - num
        
        // 在哈希表中查找 complement
        if j, ok := numMap[complement]; ok && j != i {
            // 找到目标和,返回两个元素的下标
            return []int{j, i}
        }
        
        // 将当前元素的值和下标存入哈希表
        numMap[num] = i
    }
    
    // 没有找到目标和,返回空切片
    return []int{}
}
```
