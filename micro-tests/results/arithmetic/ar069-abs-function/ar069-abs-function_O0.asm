
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar069-abs-function/ar069-abs-function_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_absi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe8    	ldr	w8, [sp, #0xc]
10000036c: 36f800e8    	tbz	w8, #0x1f, 0x100000388 <__Z8test_absi+0x28>
100000370: 14000001    	b	0x100000374 <__Z8test_absi+0x14>
100000374: b9400fe9    	ldr	w9, [sp, #0xc]
100000378: 52800008    	mov	w8, #0x0                ; =0
10000037c: 6b090108    	subs	w8, w8, w9
100000380: b9000be8    	str	w8, [sp, #0x8]
100000384: 14000004    	b	0x100000394 <__Z8test_absi+0x34>
100000388: b9400fe8    	ldr	w8, [sp, #0xc]
10000038c: b9000be8    	str	w8, [sp, #0x8]
100000390: 14000001    	b	0x100000394 <__Z8test_absi+0x34>
100000394: b9400be0    	ldr	w0, [sp, #0x8]
100000398: 910043ff    	add	sp, sp, #0x10
10000039c: d65f03c0    	ret

00000001000003a0 <_main>:
1000003a0: d10083ff    	sub	sp, sp, #0x20
1000003a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a8: 910043fd    	add	x29, sp, #0x10
1000003ac: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b0: 12800080    	mov	w0, #-0x5               ; =-5
1000003b4: 97ffffeb    	bl	0x100000360 <__Z8test_absi>
1000003b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003bc: 910083ff    	add	sp, sp, #0x20
1000003c0: d65f03c0    	ret
