
/Users/jim/work/cppfort/micro-tests/results/memory/mem040-pointer-aliasing/mem040-pointer-aliasing_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z21test_pointer_aliasingv>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: 910073e8    	add	x8, sp, #0x1c
100000368: 52800149    	mov	w9, #0xa                ; =10
10000036c: b9001fe9    	str	w9, [sp, #0x1c]
100000370: aa0803e9    	mov	x9, x8
100000374: f9000be9    	str	x9, [sp, #0x10]
100000378: f90007e8    	str	x8, [sp, #0x8]
10000037c: f9400be9    	ldr	x9, [sp, #0x10]
100000380: 52800288    	mov	w8, #0x14               ; =20
100000384: b9000128    	str	w8, [x9]
100000388: f94007e8    	ldr	x8, [sp, #0x8]
10000038c: b9400100    	ldr	w0, [x8]
100000390: 910083ff    	add	sp, sp, #0x20
100000394: d65f03c0    	ret

0000000100000398 <_main>:
100000398: d10083ff    	sub	sp, sp, #0x20
10000039c: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a0: 910043fd    	add	x29, sp, #0x10
1000003a4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a8: 97ffffee    	bl	0x100000360 <__Z21test_pointer_aliasingv>
1000003ac: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b0: 910083ff    	add	sp, sp, #0x20
1000003b4: d65f03c0    	ret
