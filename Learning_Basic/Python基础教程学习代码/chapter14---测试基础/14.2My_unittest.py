# Author:Zhang Yuan

def square(x):
    if x==7:
        return "Bug!"
    else:
        return x*x

import unittest

class ProductTestCase(unittest.TestCase):
    def test_integers(self):
        for x in range(-10,10):
            p=square(x)
            self.assertEqual(p,x*x,"Integer Failed")
    def test_floats(self):
        for x in range(-10,10):
            x=x/10
            p = square(x)
            self.assertEqual(p, x * x, "Float Failed")

if __name__=="__main__":
    # 这个函数表示：实例化所有的TestCase子类，并运行所有名称以test打头的方法。
    unittest.main()

