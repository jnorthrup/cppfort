
/Users/jim/work/cppfort/micro-tests/results/memory/mem043-offsetof-pattern/mem043-offsetof-pattern_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z13test_offsetofv>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: 90000009    	adrp	x9, 0x100000000
1000003b8: 91104129    	add	x9, x9, #0x410
1000003bc: f940012a    	ldr	x10, [x9]
1000003c0: 910043e8    	add	x8, sp, #0x10
1000003c4: f9000bea    	str	x10, [sp, #0x10]
1000003c8: b9400929    	ldr	w9, [x9, #0x8]
1000003cc: b9001be9    	str	w9, [sp, #0x18]
1000003d0: f90007e8    	str	x8, [sp, #0x8]
1000003d4: f94007e8    	ldr	x8, [sp, #0x8]
1000003d8: 91001108    	add	x8, x8, #0x4
1000003dc: f90003e8    	str	x8, [sp]
1000003e0: f94003e8    	ldr	x8, [sp]
1000003e4: b9400100    	ldr	w0, [x8]
1000003e8: 910083ff    	add	sp, sp, #0x20
1000003ec: d65f03c0    	ret

00000001000003f0 <_main>:
1000003f0: d10083ff    	sub	sp, sp, #0x20
1000003f4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003f8: 910043fd    	add	x29, sp, #0x10
1000003fc: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000400: 97ffffec    	bl	0x1000003b0 <__Z13test_offsetofv>
100000404: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000408: 910083ff    	add	sp, sp, #0x20
10000040c: d65f03c0    	ret
