
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf048-do-while-assignment/cf048-do-while-assignment_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9get_valueRi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007e0    	str	x0, [sp, #0x8]
100000368: f94007e9    	ldr	x9, [sp, #0x8]
10000036c: b9400120    	ldr	w0, [x9]
100000370: 11000408    	add	w8, w0, #0x1
100000374: b9000128    	str	w8, [x9]
100000378: 910043ff    	add	sp, sp, #0x10
10000037c: d65f03c0    	ret

0000000100000380 <__Z24test_do_while_assignmentv>:
100000380: d10083ff    	sub	sp, sp, #0x20
100000384: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000388: 910043fd    	add	x29, sp, #0x10
10000038c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000390: b9000bff    	str	wzr, [sp, #0x8]
100000394: 14000001    	b	0x100000398 <__Z24test_do_while_assignmentv+0x18>
100000398: d10013a0    	sub	x0, x29, #0x4
10000039c: 97fffff1    	bl	0x100000360 <__Z9get_valueRi>
1000003a0: b90007e0    	str	w0, [sp, #0x4]
1000003a4: b94007e9    	ldr	w9, [sp, #0x4]
1000003a8: b9400be8    	ldr	w8, [sp, #0x8]
1000003ac: 0b090108    	add	w8, w8, w9
1000003b0: b9000be8    	str	w8, [sp, #0x8]
1000003b4: 14000001    	b	0x1000003b8 <__Z24test_do_while_assignmentv+0x38>
1000003b8: b94007e8    	ldr	w8, [sp, #0x4]
1000003bc: 71002908    	subs	w8, w8, #0xa
1000003c0: 54fffecb    	b.lt	0x100000398 <__Z24test_do_while_assignmentv+0x18>
1000003c4: 14000001    	b	0x1000003c8 <__Z24test_do_while_assignmentv+0x48>
1000003c8: b9400be0    	ldr	w0, [sp, #0x8]
1000003cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d0: 910083ff    	add	sp, sp, #0x20
1000003d4: d65f03c0    	ret

00000001000003d8 <_main>:
1000003d8: d10083ff    	sub	sp, sp, #0x20
1000003dc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e0: 910043fd    	add	x29, sp, #0x10
1000003e4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e8: 97ffffe6    	bl	0x100000380 <__Z24test_do_while_assignmentv>
1000003ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret
