
import os

NVCOMP = "../../nvcomp/build"

# First compile the program
out = os.popen("nvcc lz4_via_nvcomp.cu -I %s/include -L %s/lib -lnvcomp -o lz4_via_nvcomp"%(NVCOMP, NVCOMP) ).read()
print(out)

for bits in 8,16,32:
    stem = "Primes_u%d_uniform15"%(bits)
    out = os.popen("LD_LIBRARY_PATH=%s/lib nvprof ./lz4_via_nvcomp %s %d %s"%(
        NVCOMP, stem + ".bslz4", bits//8, stem + ".bs_nvcomp") ).read()
    print(out)
    ref = open(stem+".bs","rb").read()
    cprog = open(stem+".bs_nvcomp","rb").read()
    assert ref == cprog
    print("C program matched python version")
