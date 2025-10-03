
/Users/jim/work/cppfort/micro-tests/results/memory/mem027-void-pointer/mem027-void-pointer_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_void_pointerv>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: 910073e8    	add	x8, sp, #0x1c
100000368: 52800549    	mov	w9, #0x2a               ; =42
10000036c: b9001fe9    	str	w9, [sp, #0x1c]
100000370: f9000be8    	str	x8, [sp, #0x10]
100000374: f9400be8    	ldr	x8, [sp, #0x10]
100000378: f90007e8    	str	x8, [sp, #0x8]
10000037c: f94007e8    	ldr	x8, [sp, #0x8]
100000380: b9400100    	ldr	w0, [x8]
100000384: 910083ff    	add	sp, sp, #0x20
100000388: d65f03c0    	ret

000000010000038c <_main>:
10000038c: d10083ff    	sub	sp, sp, #0x20
100000390: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000394: 910043fd    	add	x29, sp, #0x10
100000398: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000039c: 97fffff1    	bl	0x100000360 <__Z17test_void_pointerv>
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret
