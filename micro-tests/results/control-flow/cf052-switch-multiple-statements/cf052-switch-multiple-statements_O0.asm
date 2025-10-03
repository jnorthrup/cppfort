
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf052-switch-multiple-statements/cf052-switch-multiple-statements_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_switch_multi_stmti>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b90007e8    	str	w8, [sp, #0x4]
100000374: 71000508    	subs	w8, w8, #0x1
100000378: 540000c0    	b.eq	0x100000390 <__Z22test_switch_multi_stmti+0x30>
10000037c: 14000001    	b	0x100000380 <__Z22test_switch_multi_stmti+0x20>
100000380: b94007e8    	ldr	w8, [sp, #0x4]
100000384: 71000908    	subs	w8, w8, #0x2
100000388: 54000160    	b.eq	0x1000003b4 <__Z22test_switch_multi_stmti+0x54>
10000038c: 14000011    	b	0x1000003d0 <__Z22test_switch_multi_stmti+0x70>
100000390: 528000a8    	mov	w8, #0x5                ; =5
100000394: b9000be8    	str	w8, [sp, #0x8]
100000398: b9400be8    	ldr	w8, [sp, #0x8]
10000039c: 531f7908    	lsl	w8, w8, #1
1000003a0: b9000be8    	str	w8, [sp, #0x8]
1000003a4: b9400be8    	ldr	w8, [sp, #0x8]
1000003a8: 11000d08    	add	w8, w8, #0x3
1000003ac: b9000be8    	str	w8, [sp, #0x8]
1000003b0: 1400000a    	b	0x1000003d8 <__Z22test_switch_multi_stmti+0x78>
1000003b4: 52800148    	mov	w8, #0xa                ; =10
1000003b8: b9000be8    	str	w8, [sp, #0x8]
1000003bc: b9400be8    	ldr	w8, [sp, #0x8]
1000003c0: 52800049    	mov	w9, #0x2                ; =2
1000003c4: 1ac90d08    	sdiv	w8, w8, w9
1000003c8: b9000be8    	str	w8, [sp, #0x8]
1000003cc: 14000003    	b	0x1000003d8 <__Z22test_switch_multi_stmti+0x78>
1000003d0: b9000bff    	str	wzr, [sp, #0x8]
1000003d4: 14000001    	b	0x1000003d8 <__Z22test_switch_multi_stmti+0x78>
1000003d8: b9400be0    	ldr	w0, [sp, #0x8]
1000003dc: 910043ff    	add	sp, sp, #0x10
1000003e0: d65f03c0    	ret

00000001000003e4 <_main>:
1000003e4: d10083ff    	sub	sp, sp, #0x20
1000003e8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003ec: 910043fd    	add	x29, sp, #0x10
1000003f0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003f4: 52800020    	mov	w0, #0x1                ; =1
1000003f8: 97ffffda    	bl	0x100000360 <__Z22test_switch_multi_stmti>
1000003fc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000400: 910083ff    	add	sp, sp, #0x20
100000404: d65f03c0    	ret
