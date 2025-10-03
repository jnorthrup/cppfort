
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar024-div-double/ar024-div-double_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_div_doubledd>:
100000360: 1e611800    	fdiv	d0, d0, d1
100000364: 1e602028    	fcmp	d1, #0.0
100000368: 2f00e401    	movi	d1, #0000000000000000
10000036c: 1e600c20    	fcsel	d0, d1, d0, eq
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 528000a0    	mov	w0, #0x5                ; =5
100000378: d65f03c0    	ret
