# 关于python并发的理解

## 1.   多线程Threading

### 创建线程

​    thread = threading.Thread(target=thread_job,)   # 定义线程 

thread.start()  # 让线程开始工作

thread.join()   #阻塞线程，控制线程进度，等待线程执行完毕后才执行join下面的代码

 

获取已激活的线程数

threading.active_count()

查看现在正在运行的线程

threading.current_thread()

### 储存进程结果 Queue

在多线程函数中定义一个Queue，用来保存返回值，代替return，定义一个多线程列表，初始化一个多维数据列表，用来处理：

 

def multithreading():

​    q =Queue()    #q中存放返回值，代替return的返回值

​    threads = []

​    data = [[1,2,3],[3,4,5],[4,4,4],[5,5,5]]

在多线程函数中定义四个线程，启动线程，将每个线程添加到多线程的列表中

 

for i in range(4):   #定义四个线程

​    t = threading.Thread(target=job,args=(data[i],q)) #Thread首字母要大写，被调用的job函数没有括号，只是一个索引，参数在后面

​    t.start()#开始线程

​    threads.append(t) #把每个线程append到线程列表中

分别join四个线程到主线程

 

for thread in threads:

​    thread.join()

定义一个空的列表results，将四个线运行后保存在队列中的结果返回给空列表results

 

results = []

for _ in range(4):

​    results.append(q.get())  #q.get()按顺序从q中拿出一个值

print(results)

**完整的代码** 

import threading

import time

 

from queue import Queue

 

def job(l,q):

​    for i in range (len(l)):

​        l[i] = l[i]**2

​    q.put(l)

 

def multithreading():

​    q =Queue()

​    threads = []

​    data = [[1,2,3],[3,4,5],[4,4,4],[5,5,5]]

​    for i in range(4):

​        t = threading.Thread(target=job,args=(data[i],q))

​        t.start()

​        threads.append(t)

​    for thread in threads:

​        thread.join()

​    results = []

​    for _ in range(4):

​        results.append(q.get())

​    print(results)

 

if __name___=='__main__':

​    multithreading()

最后运行结果为:

 

[[1, 4, 9], [9, 16, 25], [16, 16, 16], [25, 25, 25]]

### GIL

python 的多线程 threading 有时候并不是特别理想. 最主要的原因是就是, Python 的设计上, 有一个必要的环节, 就是 Global Interpreter Lock (GIL). 这个东西让 Python 还是一次性只能处理一个东西.

 

我从这里摘抄了一段对于 GIL 的解释.

 

尽管Python完全支持多线程编程， 但是解释器的C语言实现部分在完全并行执行时并不是线程安全的。 实际上，解释器被一个全局解释器锁保护着，它确保任何时候都只有一个Python线程执行。 GIL最大的问题就是Python的多线程程序并不能利用多核CPU的优势 （比如一个使用了多个线程的计算密集型程序只会在一个单CPU上面运行）。

 

在讨论普通的GIL之前，有一点要强调的是GIL只会影响到那些严重依赖CPU的程序（比如计算型的）。 如果你的程序大部分只会涉及到I/O，比如网络交互，那么使用多线程就很合适， 因为它们大部分时间都在等待。实际上，你完全可以放心的创建几千个Python线程， 现代操作系统运行这么多线程没有任何压力，没啥可担心的。

### 线程锁

lock在不同线程使用同一共享内存时，能够确保线程之间互不影响，使用lock的方法是， 在每个线程执行运算修改共享内存之前，执行lock.acquire()将共享内存上锁， 确保当前线程执行时，内存不会被其他线程访问，执行运算完毕后，使用lock.release()将锁打开， 保证其他的线程可以使用该共享内存。

函数一和函数二加锁

import threading

 

def job1():

​    global A,lock

​    lock.acquire()

​    for i in range(10):

​        A+=1

​        print('job1',A)

​    lock.release()

 

def job2():

​    global A,lock

​    lock.acquire()

​    for i in range(10):

​        A+=10

​        print('job2',A)

​    lock.release()

 

if __name__== '__main__':

​    lock=threading.Lock()

​    A=0

​    t1=threading.Thread(target=job1)

​    t2=threading.Thread(target=job2)

​    t1.start()

​    t2.start()

​    t1.join()

​    t2.join()

## 2.   多进程Multiprocessing

### 创建进程与创建线程类似

import multiprocessing as mp

import threading as td

 

def job(a,d):

​    print('aaaaa')

 

t1 = td.Thread(target=job,args=(1,2))

p1 = mp.Process(target=job,args=(1,2))

t1.start()

p1.start()

t1.join()

p1.join()

### Queue保存进程运行结果和进程间传递数据

import multiprocessing as mp

 

def job(q):

​    res=q.get()

​    for i in range(10):

​        res+=1

​        print('job:{}'.format(res))

q.put(res)    

 

def job2(q):

​    res=q.get()

​    for i in range(10):

​        res+=2

​        print('job2:{}'.format(res))

​    q.put(res)   

 

if __name__=='__main__':

​    q = mp.Queue()

​    q.put(0)

​    p1 = mp.Process(target=job,args=(q,))

​    p2 = mp.Process(target=job2,args=(q,))

​    p1.start()

​    p2.start()

​    p1.join()

​    p2.join()

​    res1 = q.get()

​    print('res1 %d' % res1)

   

### 进程池

**进程池 Pool() 和 map()** 

然后我们定义一个Pool

 

pool = mp.Pool()

有了池子之后，就可以让池子对应某一个函数，我们向池子里丢数据，池子就会返回函数返回的值。 Pool和之前的Process的不同点是丢向Pool的函数有返回值，而Process的没有返回值。

 

接下来用map()获取结果，在map()中需要放入函数和需要迭代运算的值，然后它会自动分配给CPU核，返回结果

 

res = pool.map(job, range(10))

让我们来运行一下

 

def multicore():

​    pool = mp.Pool()

​    res = pool.map(job, range(10))

​    print(res)

​    

if __name__ == '__main__':

​    multicore()

运行结果：

 

[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

 

**apply_async()** 

Pool除了map()外，还有可以返回结果的方式，那就是apply_async().

 

apply_async()中只能传递一个值，它只会放入一个核进行运算，但是传入值时要注意是可迭代的，所以在传入值后需要加逗号, 同时需要用get()方法获取返回值

 

def multicore():

​    pool = mp.Pool() 

​    res = pool.map(job, range(10))

​    print(res)

​    res = pool.apply_async(job, (2,))

​    \# 用get获得结果

​    print(res.get())

运行结果；

4 # apply_async()

### 进程锁

import multiprocessing as mp  

  l = mp.Lock() # 定义一个进程锁

​    l.acquire() # 锁住

​    l.release() # 释放

## 3.   协程gevent，asyncio，yeild

### 概念

　　协程，又称微线程，纤程，英文名Coroutine。协程的作用，是在执行函数A时，可以随时中断，去执行函数B，然后中断继续执行函数A（可以自由切换）。但这一过程并不是函数调用（没有调用语句），这一整个过程看似像多线程，然而协程只有一个线程执行。

### 优势

执行效率极高，因为子程序切换（函数）不是线程切换，由程序自身控制，没有切换线程的开销。所以与多线程相比，线程的数量越多，协程性能的优势越明显。

不需要多线程的锁机制，因为只有一个线程，也不存在同时写变量冲突，在控制共享资源时也不需要加锁，因此执行效率高很多。

　　说明：协程可以处理IO密集型程序的效率问题，但是处理CPU密集型不是它的长处，如要充分发挥CPU利用率可以结合多进程+协程。

 

　　以上只是协程的一些概念，可能听起来比较抽象，那么我结合代码讲一讲吧。这里主要介绍协程在Python的应用，Python2对协程的支持比较有限，生成器的yield实现了一部分但不完全，gevent模块倒是有比较好的实现；Python3.4以后引入了asyncio模块，可以很好的使用协程。

### 安装gevent模块

pip3 install gevent

### Gevent实例

import gevent

import requests

from gevent **import monkey**

\# socket发送请求以后就会进入等待状态，打上猴子补丁后gevent更改了这个机制。

\# socket.setblocking(False)  -->发送请求后就不会等待服务器响应

**monkey.patch_all()**  #猴子补丁 找到内置的socket并更改为gevent自己的东西

 

def fetch_async(method, url, req_kwargs):

​    print(method, url, req_kwargs)

​    response = requests.request(method=method, url=url, **req_kwargs)

​    print(response.url, response.content)

 

\# ##### 发送请求 #####

gevent.joinall([

​    \# 这里spawn是3个任务[实际是3个协程]，每个任务都会执行fetch_async函数

​    gevent.spawn(fetch_async, method='get', url='https://www.python.org/', req_kwargs={}),

​    gevent.spawn(fetch_async, method='get', url='https://www.yahoo.com/', req_kwargs={}),

​    gevent.spawn(fetch_async, method='get', url='https://github.com/', req_kwargs={}),

])

 

### Gevent也是支持协程池

\##### 发送请求（协程池控制最大协程数量） #####

\# 也可以理解为先最大发送2个请求，2个请求结束后发送第三个请求

from gevent.pool import Pool

pool = Pool(2)  # 最多执行2个协程序，None表示不设置限制

gevent.joinall([

​    pool.spawn(fetch_async, method='get', url='https://www.python.org/', req_kwargs={}),

​    pool.spawn(fetch_async, method='get', url='https://www.yahoo.com/', req_kwargs={}),

​    pool.spawn(fetch_async, method='get', url='https://www.github.com/', req_kwargs={}),

])

 

## 个人理解

正常情况下多线程和多进程是无法保存运行结果的，但是可以使用同一个**队列queue**对象来保存线程或者进程运行的结果。

由于python中GIL锁的存在，多线程不一定有很高效率。GIL相当于一个全局的线程锁，同一时间值允许一个线程运行，当这个线程遇到io时切换其他线程运行。这点比较像协程。

效率最高的搭配方式是：多进程+协程。多进程充分利用多核cpu性能，协程充分利用单核的性能。

gevent并不是同时执行，还是按顺序执行，并未打乱输出结果，多线程不按顺序执行，打乱了输出结果。网络状况好（IO操作延时低）的情况下，gevent能稍微提高点效率，IO操作很费时的情况gevent效率将大大提高，效率最高的还是多线程。

 

 

 

 

 

 

 

 

 