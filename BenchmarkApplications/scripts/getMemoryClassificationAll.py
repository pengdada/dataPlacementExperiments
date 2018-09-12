#!/usr/bin/env python
import collections
import sys

filename = '../doc/AllBenchmarkMemoryTypes.csv'

def main(argv):
    if(len(sys.argv) < 3):
        print("Script requires two arguments: { platform, benchmark }!")
        exit(1)
    else:
        PLATFORM= str(sys.argv[1])
        BENCHMARK = str(sys.argv[2])

    d = collections.defaultdict(dict)

    with open(filename) as fp:
        for line in fp:
            (platform, benchmark, classf) = line.split(",")
            d[platform][benchmark] = classf

    for platform,nameClass in d.items():
        if(platform.strip() == PLATFORM.strip()):
            for benchmark,c in nameClass.items():
                if(benchmark.strip() == BENCHMARK.strip()):
                    print(c.rstrip())
                    exit(0)
    print("NA")

if __name__ == "__main__":
   main(sys.argv[1:])
