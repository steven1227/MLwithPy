__author__ = 'steven'
import sys, os

test = 'steven'
# from Ex1 import ex1

number = 23
a = [1, 2, 3]
i = 1234

print os.getcwd()


def compute_cost(x=None):
    b = 123
    for i in range(3):
        print i
    print str(i) + " hi,123"


def func(a=1, **b):
    """Print

        detail."""
    print b

#
# print func(c="test", b="rendong")
print func.__doc__
x = int(1)
y = int(3)
print y.__init__(x=2)
y = {(1, 2): 2, (3, 5): 4}

a = (1, 2)
b = (1, 2)
# print id(a), id(b)
c = [1, 2]
d = [1, 2]
a = [2222, c, 2, 3]
name = 10000


class Person:
    name = 12.33
    name += 1

    def __init__(self, name):
        name = 123
        self.__hide = 12344
        self.name = name

    def test(self):
        print "just a test", Person.name, self.__hide

    @classmethod
    def say_hi(cls):
        global name
        name = 12.34
        cls.name += 5
        print name


p = Person("rendong")
p.test()
print p.__class__


class SchoolMember:
    '''Represents any school member.'''

    def __init__(self, name, age):
        self.name = name
        self.age = age
        print '(Initialized SchoolMember: {})'.format(self.name)

    def tell(self):
        '''Tell my details.'''
        print 'Name:"{}" Age:"{}"'.format(self.name, self.age),


class Teacher(SchoolMember):
    '''Represents a teacher.'''

    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print '(Initialized Teacher: {})'.format(self.tell())

        # def tell(self):
        #     SchoolMember.tell(self)
        #     print 'Salary: "{:d}"'.format(self.salary)


class Student(SchoolMember):
    '''Represents a student.'''

    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print '(Initialized Student: {})'.format(self.name)

    def tell(self):
        SchoolMember.tell(self)
        print 'Marks: "{:d}"'.format(self.marks)


t = Teacher('Mrs. Shrividya', 40, 30000)
s = Student('Swaroop', 25, 75)

# prints a blank line
print


class Parent():
    def __init__(self):
        pass

    def __init__(self, a):
        self.a = a
        print "This is parent"
        pass


class Son(Parent):
    def __init__(self):
        super(self.__class__, self).__init__()
        print "This is Son"
        pass

def is_prime(n):
    return len(filter(lambda k: n % k == 0, range(2, n))) == 0

def primes(m):
    print filter(is_prime, range(1, m))


print primes(10)
