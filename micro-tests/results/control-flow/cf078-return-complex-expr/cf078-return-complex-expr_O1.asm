
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf078-return-complex-expr/cf078-return-complex-expr_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z7computeii>:
100000360: 0b000028    	add	w8, w1, w0
100000364: 4b010009    	sub	w9, w0, w1
100000368: 1b007c2a    	mul	w10, w1, w0
10000036c: 1b092900    	madd	w0, w8, w9, w10
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 528003e0    	mov	w0, #0x1f               ; =31
100000378: d65f03c0    	ret
