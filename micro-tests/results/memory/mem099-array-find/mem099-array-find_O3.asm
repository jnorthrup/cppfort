
/Users/jim/work/cppfort/micro-tests/results/memory/mem099-array-find/mem099-array-find_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_array_findi>:
100000360: 51000408    	sub	w8, w0, #0x1
100000364: 7100151f    	cmp	w8, #0x5
100000368: 5a9f3100    	csinv	w0, w8, wzr, lo
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800040    	mov	w0, #0x2                ; =2
100000374: d65f03c0    	ret
