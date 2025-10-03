
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar062-three-way-compare/ar062-three-way-compare_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_three_wayii>:
100000360: 12800008    	mov	w8, #-0x1               ; =-1
100000364: 6b01001f    	cmp	w0, w1
100000368: 5a88b508    	cneg	w8, w8, ge
10000036c: 1a8803e0    	csel	w0, wzr, w8, eq
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800020    	mov	w0, #0x1                ; =1
100000378: d65f03c0    	ret
