
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar070-sign-function/ar070-sign-function_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9test_signi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 71000108    	subs	w8, w8, #0x0
100000370: 540000ad    	b.le	0x100000384 <__Z9test_signi+0x24>
100000374: 14000001    	b	0x100000378 <__Z9test_signi+0x18>
100000378: 52800028    	mov	w8, #0x1                ; =1
10000037c: b9000fe8    	str	w8, [sp, #0xc]
100000380: 14000009    	b	0x1000003a4 <__Z9test_signi+0x44>
100000384: b9400be8    	ldr	w8, [sp, #0x8]
100000388: 36f800a8    	tbz	w8, #0x1f, 0x10000039c <__Z9test_signi+0x3c>
10000038c: 14000001    	b	0x100000390 <__Z9test_signi+0x30>
100000390: 12800008    	mov	w8, #-0x1               ; =-1
100000394: b9000fe8    	str	w8, [sp, #0xc]
100000398: 14000003    	b	0x1000003a4 <__Z9test_signi+0x44>
10000039c: b9000fff    	str	wzr, [sp, #0xc]
1000003a0: 14000001    	b	0x1000003a4 <__Z9test_signi+0x44>
1000003a4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003a8: 910043ff    	add	sp, sp, #0x10
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b8: 910043fd    	add	x29, sp, #0x10
1000003bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c0: 12800080    	mov	w0, #-0x5               ; =-5
1000003c4: 97ffffe7    	bl	0x100000360 <__Z9test_signi>
1000003c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003cc: 910083ff    	add	sp, sp, #0x20
1000003d0: d65f03c0    	ret
