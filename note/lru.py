class DoubleLinkedList:
    class Node:
        def __init__(self, data, pre=None, next=None):
            self.pre = pre
            self.next = next
            self.data = data

    def __init__(self):
        self.root = DoubleLinkedList.Node(data="_root_node")
        self.tail = self.root

    def print(self):
        data_list = []
        cur = self.root
        while cur is not None:
            data_list.append(str(cur.data))
            cur = cur.next
        return ", ".join(data_list)

    def push_left(self, data):
        node = DoubleLinkedList.Node(data=data, pre=self.root)
        if self.root.next is not None:
            next_ = self.root.next
            node.next = next_
            next_.pre = node
        else:
            self.tail = node
        self.root.next = node
        return node

    def pop(self, node):
        assert node.pre is not None
        pre = node.pre
        next_ = node.next
        pre.next = next_
        if next_ is not None:
            next_.pre = pre
        else:
            self.tail = pre

    def pop_right(self):
        assert self.tail.pre is not None
        pre = self.tail.pre
        pre.next = None
        tail = self.tail
        self.tail.pre = None
        self.tail = pre
        return tail


class LRU:
    def __init__(self, size: int):
        self.size = size
        self.lru = DoubleLinkedList()
        self.cache = {}

    def put(self, k, v):
        if k not in self.cache:
            if len(self.cache) == self.size:
                to_del = self.lru.pop_right()
                del self.cache[to_del.data[0]]
        else:
            self.lru.pop(self.cache[k])

        self.cache[k] = self.lru.push_left((k, v))

    def get(self, k):
        if k not in self.cache:
            return -1
        old_node = self.cache[k]
        v = old_node.data[1]
        self.lru.pop(old_node)
        self.cache[k] = self.lru.push_left(old_node.data)
        return v


if __name__ == "__main__":
    cache = LRU(3)
    cache.put(1, 1)
    cache.put(2, 2)
    print(f"2,1, {cache.lru.print()}")
    print(f"{cache.get(1)=}")
    print(f"1,2, {cache.lru.print()}")

    cache.put(3, 3)
    print(f"3,1,2, {cache.lru.print()}")

    print(f"{cache.get(2)=}")
    print(f"2,3,1, {cache.lru.print()}")

    cache.put(3, 33)
    print(f"3,2,1, {cache.lru.print()}")
