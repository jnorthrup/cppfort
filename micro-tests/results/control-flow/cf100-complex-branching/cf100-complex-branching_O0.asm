
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf100-complex-branching/cf100-complex-branching_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_complex_branchingiii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000be1    	str	w1, [sp, #0x8]
10000036c: b90007e2    	str	w2, [sp, #0x4]
100000370: b90003ff    	str	wzr, [sp]
100000374: b9400fe8    	ldr	w8, [sp, #0xc]
100000378: b9400be9    	ldr	w9, [sp, #0x8]
10000037c: 6b090108    	subs	w8, w8, w9
100000380: 540003ad    	b.le	0x1000003f4 <__Z22test_complex_branchingiii+0x94>
100000384: 14000001    	b	0x100000388 <__Z22test_complex_branchingiii+0x28>
100000388: b9400be8    	ldr	w8, [sp, #0x8]
10000038c: b94007e9    	ldr	w9, [sp, #0x4]
100000390: 6b090108    	subs	w8, w8, w9
100000394: 5400012d    	b.le	0x1000003b8 <__Z22test_complex_branchingiii+0x58>
100000398: 14000001    	b	0x10000039c <__Z22test_complex_branchingiii+0x3c>
10000039c: b9400fe8    	ldr	w8, [sp, #0xc]
1000003a0: b9400be9    	ldr	w9, [sp, #0x8]
1000003a4: 0b090108    	add	w8, w8, w9
1000003a8: b94007e9    	ldr	w9, [sp, #0x4]
1000003ac: 0b090108    	add	w8, w8, w9
1000003b0: b90003e8    	str	w8, [sp]
1000003b4: 1400000f    	b	0x1000003f0 <__Z22test_complex_branchingiii+0x90>
1000003b8: b9400fe8    	ldr	w8, [sp, #0xc]
1000003bc: b94007e9    	ldr	w9, [sp, #0x4]
1000003c0: 6b090108    	subs	w8, w8, w9
1000003c4: 540000ed    	b.le	0x1000003e0 <__Z22test_complex_branchingiii+0x80>
1000003c8: 14000001    	b	0x1000003cc <__Z22test_complex_branchingiii+0x6c>
1000003cc: b9400fe8    	ldr	w8, [sp, #0xc]
1000003d0: b94007e9    	ldr	w9, [sp, #0x4]
1000003d4: 0b090108    	add	w8, w8, w9
1000003d8: b90003e8    	str	w8, [sp]
1000003dc: 14000004    	b	0x1000003ec <__Z22test_complex_branchingiii+0x8c>
1000003e0: b94007e8    	ldr	w8, [sp, #0x4]
1000003e4: b90003e8    	str	w8, [sp]
1000003e8: 14000001    	b	0x1000003ec <__Z22test_complex_branchingiii+0x8c>
1000003ec: 14000001    	b	0x1000003f0 <__Z22test_complex_branchingiii+0x90>
1000003f0: 1400001a    	b	0x100000458 <__Z22test_complex_branchingiii+0xf8>
1000003f4: b9400fe8    	ldr	w8, [sp, #0xc]
1000003f8: b94007e9    	ldr	w9, [sp, #0x4]
1000003fc: 6b090108    	subs	w8, w8, w9
100000400: 540000ed    	b.le	0x10000041c <__Z22test_complex_branchingiii+0xbc>
100000404: 14000001    	b	0x100000408 <__Z22test_complex_branchingiii+0xa8>
100000408: b9400be8    	ldr	w8, [sp, #0x8]
10000040c: b9400fe9    	ldr	w9, [sp, #0xc]
100000410: 0b090108    	add	w8, w8, w9
100000414: b90003e8    	str	w8, [sp]
100000418: 1400000f    	b	0x100000454 <__Z22test_complex_branchingiii+0xf4>
10000041c: b9400be8    	ldr	w8, [sp, #0x8]
100000420: b94007e9    	ldr	w9, [sp, #0x4]
100000424: 6b090108    	subs	w8, w8, w9
100000428: 540000ed    	b.le	0x100000444 <__Z22test_complex_branchingiii+0xe4>
10000042c: 14000001    	b	0x100000430 <__Z22test_complex_branchingiii+0xd0>
100000430: b9400be8    	ldr	w8, [sp, #0x8]
100000434: b94007e9    	ldr	w9, [sp, #0x4]
100000438: 0b090108    	add	w8, w8, w9
10000043c: b90003e8    	str	w8, [sp]
100000440: 14000004    	b	0x100000450 <__Z22test_complex_branchingiii+0xf0>
100000444: b94007e8    	ldr	w8, [sp, #0x4]
100000448: b90003e8    	str	w8, [sp]
10000044c: 14000001    	b	0x100000450 <__Z22test_complex_branchingiii+0xf0>
100000450: 14000001    	b	0x100000454 <__Z22test_complex_branchingiii+0xf4>
100000454: 14000001    	b	0x100000458 <__Z22test_complex_branchingiii+0xf8>
100000458: b94003e0    	ldr	w0, [sp]
10000045c: 910043ff    	add	sp, sp, #0x10
100000460: d65f03c0    	ret

0000000100000464 <_main>:
100000464: d10083ff    	sub	sp, sp, #0x20
100000468: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000046c: 910043fd    	add	x29, sp, #0x10
100000470: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000474: 528000a0    	mov	w0, #0x5                ; =5
100000478: 52800141    	mov	w1, #0xa                ; =10
10000047c: 52800062    	mov	w2, #0x3                ; =3
100000480: 97ffffb8    	bl	0x100000360 <__Z22test_complex_branchingiii>
100000484: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000488: 910083ff    	add	sp, sp, #0x20
10000048c: d65f03c0    	ret
