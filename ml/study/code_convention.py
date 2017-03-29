from abc import abstractmethod

__author__ = 'wenjusun'


class FeedParser:
    _one_underscore_=1
    _one_underscore=1
    __two_underscore=2

    _prop=1

    @property
    def prop(self):
        print "getter............"
        return self._prop

    @abstractmethod
    def a_abm(self):
        print "abc???"

    @staticmethod
    def a_st_method():
        print "a static.."

    def _one(self):
        print "I am one"

    @classmethod
    def a_class_method(self):
        print "clm"

    def __two(self):
        print "I am two"

class RssFeedParser(FeedParser):
    def sub_parser(self):
        print FeedParser._one_underscore
        FeedParser._one(self)

    def sub_parser2(self):
        FeedParser.__two(self)



a=''

def method_wrapper(f):
    def new_f():
        print "entering..."
        f()
        print "leaving"
    return new_f()

@method_wrapper
def amethod():
    print "in amethod..."

def test_method():
    feedParser = FeedParser()
    # feedParser.prop=2
    print feedParser.prop

def test_it():
    feedParser = FeedParser()

    # feedParser.a_class_method()
    # FeedParser.a_class_method()
    feedParser.a_st_method()
    print FeedParser._FeedParser__two_underscore

    rss = RssFeedParser()
    rss._one()
    rss.a_class_method()
    RssFeedParser.a_class_method()

    # rss.__two()

if __name__ == "__main__":
    # test_it()
    test_method()
    # amethod()

