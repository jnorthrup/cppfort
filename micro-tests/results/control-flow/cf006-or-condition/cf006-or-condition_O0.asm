
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf006-or-condition/cf006-or-condition_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_or_conditionii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007e1    	str	w1, [sp, #0x4]
10000036c: b9400be8    	ldr	w8, [sp, #0x8]
100000370: 71002908    	subs	w8, w8, #0xa
100000374: 540000cc    	b.gt	0x10000038c <__Z17test_or_conditionii+0x2c>
100000378: 14000001    	b	0x10000037c <__Z17test_or_conditionii+0x1c>
10000037c: b94007e8    	ldr	w8, [sp, #0x4]
100000380: 71002908    	subs	w8, w8, #0xa
100000384: 540000ad    	b.le	0x100000398 <__Z17test_or_conditionii+0x38>
100000388: 14000001    	b	0x10000038c <__Z17test_or_conditionii+0x2c>
10000038c: 52800028    	mov	w8, #0x1                ; =1
100000390: b9000fe8    	str	w8, [sp, #0xc]
100000394: 14000003    	b	0x1000003a0 <__Z17test_or_conditionii+0x40>
100000398: b9000fff    	str	wzr, [sp, #0xc]
10000039c: 14000001    	b	0x1000003a0 <__Z17test_or_conditionii+0x40>
1000003a0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003a4: 910043ff    	add	sp, sp, #0x10
1000003a8: d65f03c0    	ret

00000001000003ac <_main>:
1000003ac: d10083ff    	sub	sp, sp, #0x20
1000003b0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b4: 910043fd    	add	x29, sp, #0x10
1000003b8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003bc: 528000a0    	mov	w0, #0x5                ; =5
1000003c0: 528001e1    	mov	w1, #0xf                ; =15
1000003c4: 97ffffe7    	bl	0x100000360 <__Z17test_or_conditionii>
1000003c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003cc: 910083ff    	add	sp, sp, #0x20
1000003d0: d65f03c0    	ret
