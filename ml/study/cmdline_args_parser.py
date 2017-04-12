
def optparse_demo():
    "This has been deprecated since 2.7"
    from optparse import OptionParser

    op = OptionParser()
    op.add_option('--src_file',dest='src',help='This is the source file path')
    op.add_option('--target_file')

    (opts,args) = op.parse_args()

    # print op.print_help()
    print opts,args


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dbhost',help='db server ip address')
parser.add_argument('--version', action='version', version='%(prog)s 1.0')

args = parser.parse_args()

print args
print args.dbhost



