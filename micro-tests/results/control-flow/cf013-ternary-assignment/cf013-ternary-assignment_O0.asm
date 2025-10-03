
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf013-ternary-assignment/cf013-ternary-assignment_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_ternary_assignmenti>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe8    	ldr	w8, [sp, #0xc]
10000036c: 71000108    	subs	w8, w8, #0x0
100000370: 540000cd    	b.le	0x100000388 <__Z23test_ternary_assignmenti+0x28>
100000374: 14000001    	b	0x100000378 <__Z23test_ternary_assignmenti+0x18>
100000378: b9400fe8    	ldr	w8, [sp, #0xc]
10000037c: 531f7908    	lsl	w8, w8, #1
100000380: b90007e8    	str	w8, [sp, #0x4]
100000384: 14000006    	b	0x10000039c <__Z23test_ternary_assignmenti+0x3c>
100000388: b9400fe8    	ldr	w8, [sp, #0xc]
10000038c: 52800049    	mov	w9, #0x2                ; =2
100000390: 1ac90d08    	sdiv	w8, w8, w9
100000394: b90007e8    	str	w8, [sp, #0x4]
100000398: 14000001    	b	0x10000039c <__Z23test_ternary_assignmenti+0x3c>
10000039c: b94007e8    	ldr	w8, [sp, #0x4]
1000003a0: b9000be8    	str	w8, [sp, #0x8]
1000003a4: b9400be0    	ldr	w0, [sp, #0x8]
1000003a8: 910043ff    	add	sp, sp, #0x10
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b8: 910043fd    	add	x29, sp, #0x10
1000003bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c0: 52800140    	mov	w0, #0xa                ; =10
1000003c4: 97ffffe7    	bl	0x100000360 <__Z23test_ternary_assignmenti>
1000003c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003cc: 910083ff    	add	sp, sp, #0x20
1000003d0: d65f03c0    	ret
