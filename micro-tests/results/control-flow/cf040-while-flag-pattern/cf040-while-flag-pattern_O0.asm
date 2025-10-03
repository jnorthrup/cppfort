
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf040-while-flag-pattern/cf040-while-flag-pattern_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_while_flagv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 39003fff    	strb	wzr, [sp, #0xf]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 12800008    	mov	w8, #-0x1               ; =-1
100000370: b90007e8    	str	w8, [sp, #0x4]
100000374: 14000001    	b	0x100000378 <__Z15test_while_flagv+0x18>
100000378: 39403fe8    	ldrb	w8, [sp, #0xf]
10000037c: 52800009    	mov	w9, #0x0                ; =0
100000380: b90003e9    	str	w9, [sp]
100000384: 370000e8    	tbnz	w8, #0x0, 0x1000003a0 <__Z15test_while_flagv+0x40>
100000388: 14000001    	b	0x10000038c <__Z15test_while_flagv+0x2c>
10000038c: b9400be8    	ldr	w8, [sp, #0x8]
100000390: 71002908    	subs	w8, w8, #0xa
100000394: 1a9fa7e8    	cset	w8, lt
100000398: b90003e8    	str	w8, [sp]
10000039c: 14000001    	b	0x1000003a0 <__Z15test_while_flagv+0x40>
1000003a0: b94003e8    	ldr	w8, [sp]
1000003a4: 360001e8    	tbz	w8, #0x0, 0x1000003e0 <__Z15test_while_flagv+0x80>
1000003a8: 14000001    	b	0x1000003ac <__Z15test_while_flagv+0x4c>
1000003ac: b9400be8    	ldr	w8, [sp, #0x8]
1000003b0: 71001508    	subs	w8, w8, #0x5
1000003b4: 540000e1    	b.ne	0x1000003d0 <__Z15test_while_flagv+0x70>
1000003b8: 14000001    	b	0x1000003bc <__Z15test_while_flagv+0x5c>
1000003bc: 52800028    	mov	w8, #0x1                ; =1
1000003c0: 39003fe8    	strb	w8, [sp, #0xf]
1000003c4: b9400be8    	ldr	w8, [sp, #0x8]
1000003c8: b90007e8    	str	w8, [sp, #0x4]
1000003cc: 14000001    	b	0x1000003d0 <__Z15test_while_flagv+0x70>
1000003d0: b9400be8    	ldr	w8, [sp, #0x8]
1000003d4: 11000508    	add	w8, w8, #0x1
1000003d8: b9000be8    	str	w8, [sp, #0x8]
1000003dc: 17ffffe7    	b	0x100000378 <__Z15test_while_flagv+0x18>
1000003e0: b94007e0    	ldr	w0, [sp, #0x4]
1000003e4: 910043ff    	add	sp, sp, #0x10
1000003e8: d65f03c0    	ret

00000001000003ec <_main>:
1000003ec: d10083ff    	sub	sp, sp, #0x20
1000003f0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003f4: 910043fd    	add	x29, sp, #0x10
1000003f8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003fc: 97ffffd9    	bl	0x100000360 <__Z15test_while_flagv>
100000400: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000404: 910083ff    	add	sp, sp, #0x20
100000408: d65f03c0    	ret
