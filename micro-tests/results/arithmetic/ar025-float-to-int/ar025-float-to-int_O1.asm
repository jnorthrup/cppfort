
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar025-float-to-int/ar025-float-to-int_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_float_to_intf>:
100000360: 1e380000    	fcvtzs	w0, s0
100000364: d65f03c0    	ret

0000000100000368 <_main>:
100000368: 52800060    	mov	w0, #0x3                ; =3
10000036c: d65f03c0    	ret
