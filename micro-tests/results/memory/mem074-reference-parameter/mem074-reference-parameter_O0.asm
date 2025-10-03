
/Users/jim/work/cppfort/micro-tests/results/memory/mem074-reference-parameter/mem074-reference-parameter_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9incrementRi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007e0    	str	x0, [sp, #0x8]
100000368: f94007e9    	ldr	x9, [sp, #0x8]
10000036c: b9400128    	ldr	w8, [x9]
100000370: 11000508    	add	w8, w8, #0x1
100000374: b9000128    	str	w8, [x9]
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <_main>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: 910023e0    	add	x0, sp, #0x8
100000394: 52800528    	mov	w8, #0x29               ; =41
100000398: b9000be8    	str	w8, [sp, #0x8]
10000039c: 97fffff1    	bl	0x100000360 <__Z9incrementRi>
1000003a0: b9400be0    	ldr	w0, [sp, #0x8]
1000003a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a8: 910083ff    	add	sp, sp, #0x20
1000003ac: d65f03c0    	ret
