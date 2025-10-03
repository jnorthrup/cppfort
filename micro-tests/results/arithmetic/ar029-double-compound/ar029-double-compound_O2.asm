
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar029-double-compound/ar029-double-compound_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_double_compounddd>:
100000360: 1e612800    	fadd	d0, d0, d1
100000364: 1e602800    	fadd	d0, d0, d0
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800140    	mov	w0, #0xa                ; =10
100000370: d65f03c0    	ret
