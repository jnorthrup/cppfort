
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf007-not-condition/cf007-not-condition_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_not_conditionb>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 39002fe0    	strb	w0, [sp, #0xb]
100000368: 39402fe8    	ldrb	w8, [sp, #0xb]
10000036c: 370000a8    	tbnz	w8, #0x0, 0x100000380 <__Z18test_not_conditionb+0x20>
100000370: 14000001    	b	0x100000374 <__Z18test_not_conditionb+0x14>
100000374: 52800028    	mov	w8, #0x1                ; =1
100000378: b9000fe8    	str	w8, [sp, #0xc]
10000037c: 14000003    	b	0x100000388 <__Z18test_not_conditionb+0x28>
100000380: b9000fff    	str	wzr, [sp, #0xc]
100000384: 14000001    	b	0x100000388 <__Z18test_not_conditionb+0x28>
100000388: b9400fe0    	ldr	w0, [sp, #0xc]
10000038c: 910043ff    	add	sp, sp, #0x10
100000390: d65f03c0    	ret

0000000100000394 <_main>:
100000394: d10083ff    	sub	sp, sp, #0x20
100000398: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000039c: 910043fd    	add	x29, sp, #0x10
1000003a0: 52800008    	mov	w8, #0x0                ; =0
1000003a4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a8: 12000100    	and	w0, w8, #0x1
1000003ac: 97ffffed    	bl	0x100000360 <__Z18test_not_conditionb>
1000003b0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b4: 910083ff    	add	sp, sp, #0x20
1000003b8: d65f03c0    	ret
