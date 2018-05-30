#import unittest
from TEST import TEST

def add(x, y):
    """Add function"""
    return x + y

def multiply(x, y):
    """multiply function"""
    return x * y

#class TESt(unittest.TestCase):
#    def test_add(self):
#        self.assertEqual(add(10, 5), 15)
#        self.assertEqual(add(-1, 1), 0)
#        self.assertEqual(add(-1, -1), -2)
#
#    def test_multiply(self):
#        self.assertEqual(multiply(10, 5), 50)
#        self.assertEqual(multiply(-1, 1), -1)
#        self.assertEqual(multiply(-1, -1), 1)
#
#
#if __name__ == '__main__':
#    unittest.main()



TEST.assertEqual(add(10, 5), 15)
TEST.assertEqual(add(-1, 1), 0)
TEST.assertEqual(add(-1, -1), -2)


