
/Users/jim/work/cppfort/micro-tests/results/memory/mem075-const-reference-parameter/mem075-const-reference-parameter_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9get_valueRKi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007e0    	str	x0, [sp, #0x8]
100000368: f94007e8    	ldr	x8, [sp, #0x8]
10000036c: b9400100    	ldr	w0, [x8]
100000370: 910043ff    	add	sp, sp, #0x10
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: d10083ff    	sub	sp, sp, #0x20
10000037c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000380: 910043fd    	add	x29, sp, #0x10
100000384: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000388: 910023e0    	add	x0, sp, #0x8
10000038c: 52800548    	mov	w8, #0x2a               ; =42
100000390: b9000be8    	str	w8, [sp, #0x8]
100000394: 97fffff3    	bl	0x100000360 <__Z9get_valueRKi>
100000398: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000039c: 910083ff    	add	sp, sp, #0x20
1000003a0: d65f03c0    	ret
