
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf065-goto-nested-loops/cf065-goto-nested-loops_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_goto_nestedv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 12800008    	mov	w8, #-0x1               ; =-1
100000368: b9000fe8    	str	w8, [sp, #0xc]
10000036c: b9000bff    	str	wzr, [sp, #0x8]
100000370: 14000001    	b	0x100000374 <__Z16test_goto_nestedv+0x14>
100000374: b9400be8    	ldr	w8, [sp, #0x8]
100000378: 71002908    	subs	w8, w8, #0xa
10000037c: 5400042a    	b.ge	0x100000400 <__Z16test_goto_nestedv+0xa0>
100000380: 14000001    	b	0x100000384 <__Z16test_goto_nestedv+0x24>
100000384: b90007ff    	str	wzr, [sp, #0x4]
100000388: 14000001    	b	0x10000038c <__Z16test_goto_nestedv+0x2c>
10000038c: b94007e8    	ldr	w8, [sp, #0x4]
100000390: 71002908    	subs	w8, w8, #0xa
100000394: 540002ca    	b.ge	0x1000003ec <__Z16test_goto_nestedv+0x8c>
100000398: 14000001    	b	0x10000039c <__Z16test_goto_nestedv+0x3c>
10000039c: b9400be8    	ldr	w8, [sp, #0x8]
1000003a0: 71001508    	subs	w8, w8, #0x5
1000003a4: 540001a1    	b.ne	0x1000003d8 <__Z16test_goto_nestedv+0x78>
1000003a8: 14000001    	b	0x1000003ac <__Z16test_goto_nestedv+0x4c>
1000003ac: b94007e8    	ldr	w8, [sp, #0x4]
1000003b0: 71000d08    	subs	w8, w8, #0x3
1000003b4: 54000121    	b.ne	0x1000003d8 <__Z16test_goto_nestedv+0x78>
1000003b8: 14000001    	b	0x1000003bc <__Z16test_goto_nestedv+0x5c>
1000003bc: b9400be8    	ldr	w8, [sp, #0x8]
1000003c0: 52800149    	mov	w9, #0xa                ; =10
1000003c4: 1b097d08    	mul	w8, w8, w9
1000003c8: b94007e9    	ldr	w9, [sp, #0x4]
1000003cc: 0b090108    	add	w8, w8, w9
1000003d0: b9000fe8    	str	w8, [sp, #0xc]
1000003d4: 1400000c    	b	0x100000404 <__Z16test_goto_nestedv+0xa4>
1000003d8: 14000001    	b	0x1000003dc <__Z16test_goto_nestedv+0x7c>
1000003dc: b94007e8    	ldr	w8, [sp, #0x4]
1000003e0: 11000508    	add	w8, w8, #0x1
1000003e4: b90007e8    	str	w8, [sp, #0x4]
1000003e8: 17ffffe9    	b	0x10000038c <__Z16test_goto_nestedv+0x2c>
1000003ec: 14000001    	b	0x1000003f0 <__Z16test_goto_nestedv+0x90>
1000003f0: b9400be8    	ldr	w8, [sp, #0x8]
1000003f4: 11000508    	add	w8, w8, #0x1
1000003f8: b9000be8    	str	w8, [sp, #0x8]
1000003fc: 17ffffde    	b	0x100000374 <__Z16test_goto_nestedv+0x14>
100000400: 14000001    	b	0x100000404 <__Z16test_goto_nestedv+0xa4>
100000404: b9400fe0    	ldr	w0, [sp, #0xc]
100000408: 910043ff    	add	sp, sp, #0x10
10000040c: d65f03c0    	ret

0000000100000410 <_main>:
100000410: d10083ff    	sub	sp, sp, #0x20
100000414: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000418: 910043fd    	add	x29, sp, #0x10
10000041c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000420: 97ffffd0    	bl	0x100000360 <__Z16test_goto_nestedv>
100000424: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000428: 910083ff    	add	sp, sp, #0x20
10000042c: d65f03c0    	ret
