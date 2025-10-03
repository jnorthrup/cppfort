
/Users/jim/work/cppfort/micro-tests/results/memory/mem024-double-pointer/mem024-double-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_double_pointerv>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: 910073e9    	add	x9, sp, #0x1c
100000368: 52800548    	mov	w8, #0x2a               ; =42
10000036c: b9001fe8    	str	w8, [sp, #0x1c]
100000370: 910043e8    	add	x8, sp, #0x10
100000374: f9000be9    	str	x9, [sp, #0x10]
100000378: f90007e8    	str	x8, [sp, #0x8]
10000037c: f94007e8    	ldr	x8, [sp, #0x8]
100000380: f9400108    	ldr	x8, [x8]
100000384: b9400100    	ldr	w0, [x8]
100000388: 910083ff    	add	sp, sp, #0x20
10000038c: d65f03c0    	ret

0000000100000390 <_main>:
100000390: d10083ff    	sub	sp, sp, #0x20
100000394: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000398: 910043fd    	add	x29, sp, #0x10
10000039c: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a0: 97fffff0    	bl	0x100000360 <__Z19test_double_pointerv>
1000003a4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a8: 910083ff    	add	sp, sp, #0x20
1000003ac: d65f03c0    	ret
