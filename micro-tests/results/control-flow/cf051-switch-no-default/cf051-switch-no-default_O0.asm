
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf051-switch-no-default/cf051-switch-no-default_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_switch_no_defaulti>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: 12800008    	mov	w8, #-0x1               ; =-1
10000036c: b9000be8    	str	w8, [sp, #0x8]
100000370: b9400fe8    	ldr	w8, [sp, #0xc]
100000374: b90007e8    	str	w8, [sp, #0x4]
100000378: 71000508    	subs	w8, w8, #0x1
10000037c: 540000c0    	b.eq	0x100000394 <__Z22test_switch_no_defaulti+0x34>
100000380: 14000001    	b	0x100000384 <__Z22test_switch_no_defaulti+0x24>
100000384: b94007e8    	ldr	w8, [sp, #0x4]
100000388: 71000908    	subs	w8, w8, #0x2
10000038c: 540000a0    	b.eq	0x1000003a0 <__Z22test_switch_no_defaulti+0x40>
100000390: 14000007    	b	0x1000003ac <__Z22test_switch_no_defaulti+0x4c>
100000394: 52800148    	mov	w8, #0xa                ; =10
100000398: b9000be8    	str	w8, [sp, #0x8]
10000039c: 14000004    	b	0x1000003ac <__Z22test_switch_no_defaulti+0x4c>
1000003a0: 52800288    	mov	w8, #0x14               ; =20
1000003a4: b9000be8    	str	w8, [sp, #0x8]
1000003a8: 14000001    	b	0x1000003ac <__Z22test_switch_no_defaulti+0x4c>
1000003ac: b9400be0    	ldr	w0, [sp, #0x8]
1000003b0: 910043ff    	add	sp, sp, #0x10
1000003b4: d65f03c0    	ret

00000001000003b8 <_main>:
1000003b8: d10083ff    	sub	sp, sp, #0x20
1000003bc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c0: 910043fd    	add	x29, sp, #0x10
1000003c4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c8: 52800020    	mov	w0, #0x1                ; =1
1000003cc: 97ffffe5    	bl	0x100000360 <__Z22test_switch_no_defaulti>
1000003d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d4: 910083ff    	add	sp, sp, #0x20
1000003d8: d65f03c0    	ret
