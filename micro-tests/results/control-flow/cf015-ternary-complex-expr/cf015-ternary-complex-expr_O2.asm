
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf015-ternary-complex-expr/cf015-ternary-complex-expr_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_ternary_complexii>:
100000360: 6b01001f    	cmp	w0, w1
100000364: 5a81c428    	cneg	w8, w1, le
100000368: 0b000100    	add	w0, w8, w0
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 528001a0    	mov	w0, #0xd                ; =13
100000374: d65f03c0    	ret
