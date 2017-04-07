
class Foo():
    def get_name(self):
        return self.name

    def set_name(self,new_name):
        self.name = new_name
        return self

foo = Foo()

_=foo.set_name('Haha')

print foo.get_name()