
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf059-switch-empty-case/cf059-switch-empty-case_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_switch_emptyi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: b90007e8    	str	w8, [sp, #0x4]
100000374: 71000508    	subs	w8, w8, #0x1
100000378: 71000908    	subs	w8, w8, #0x2
10000037c: 540000e9    	b.ls	0x100000398 <__Z17test_switch_emptyi+0x38>
100000380: 14000001    	b	0x100000384 <__Z17test_switch_emptyi+0x24>
100000384: b94007e8    	ldr	w8, [sp, #0x4]
100000388: 71001108    	subs	w8, w8, #0x4
10000038c: 71000508    	subs	w8, w8, #0x1
100000390: 540000a9    	b.ls	0x1000003a4 <__Z17test_switch_emptyi+0x44>
100000394: 14000007    	b	0x1000003b0 <__Z17test_switch_emptyi+0x50>
100000398: 52800f68    	mov	w8, #0x7b               ; =123
10000039c: b9000be8    	str	w8, [sp, #0x8]
1000003a0: 14000006    	b	0x1000003b8 <__Z17test_switch_emptyi+0x58>
1000003a4: 528005a8    	mov	w8, #0x2d               ; =45
1000003a8: b9000be8    	str	w8, [sp, #0x8]
1000003ac: 14000003    	b	0x1000003b8 <__Z17test_switch_emptyi+0x58>
1000003b0: b9000bff    	str	wzr, [sp, #0x8]
1000003b4: 14000001    	b	0x1000003b8 <__Z17test_switch_emptyi+0x58>
1000003b8: b9400be0    	ldr	w0, [sp, #0x8]
1000003bc: 910043ff    	add	sp, sp, #0x10
1000003c0: d65f03c0    	ret

00000001000003c4 <_main>:
1000003c4: d10083ff    	sub	sp, sp, #0x20
1000003c8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003cc: 910043fd    	add	x29, sp, #0x10
1000003d0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d4: 52800040    	mov	w0, #0x2                ; =2
1000003d8: 97ffffe2    	bl	0x100000360 <__Z17test_switch_emptyi>
1000003dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e0: 910083ff    	add	sp, sp, #0x20
1000003e4: d65f03c0    	ret
