
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar071-logical-and/ar071-logical-and_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_logical_andbb>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 39003fe0    	strb	w0, [sp, #0xf]
100000368: 39003be1    	strb	w1, [sp, #0xe]
10000036c: 39403fe8    	ldrb	w8, [sp, #0xf]
100000370: 52800009    	mov	w9, #0x0                ; =0
100000374: b9000be9    	str	w9, [sp, #0x8]
100000378: 360000a8    	tbz	w8, #0x0, 0x10000038c <__Z16test_logical_andbb+0x2c>
10000037c: 14000001    	b	0x100000380 <__Z16test_logical_andbb+0x20>
100000380: 39403be8    	ldrb	w8, [sp, #0xe]
100000384: b9000be8    	str	w8, [sp, #0x8]
100000388: 14000001    	b	0x10000038c <__Z16test_logical_andbb+0x2c>
10000038c: b9400be8    	ldr	w8, [sp, #0x8]
100000390: 12000100    	and	w0, w8, #0x1
100000394: 910043ff    	add	sp, sp, #0x10
100000398: d65f03c0    	ret

000000010000039c <_main>:
10000039c: d10083ff    	sub	sp, sp, #0x20
1000003a0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a4: 910043fd    	add	x29, sp, #0x10
1000003a8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ac: 52800028    	mov	w8, #0x1                ; =1
1000003b0: 12000100    	and	w0, w8, #0x1
1000003b4: 12000101    	and	w1, w8, #0x1
1000003b8: 97ffffea    	bl	0x100000360 <__Z16test_logical_andbb>
1000003bc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c0: 910083ff    	add	sp, sp, #0x20
1000003c4: d65f03c0    	ret
