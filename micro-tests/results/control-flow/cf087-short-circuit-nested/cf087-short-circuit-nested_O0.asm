
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf087-short-circuit-nested/cf087-short-circuit-nested_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z25test_nested_short_circuitiii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007e1    	str	w1, [sp, #0x4]
10000036c: b90003e2    	str	w2, [sp]
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71000108    	subs	w8, w8, #0x0
100000378: 540000cd    	b.le	0x100000390 <__Z25test_nested_short_circuitiii+0x30>
10000037c: 14000001    	b	0x100000380 <__Z25test_nested_short_circuitiii+0x20>
100000380: b94007e8    	ldr	w8, [sp, #0x4]
100000384: 71000108    	subs	w8, w8, #0x0
100000388: 540000cc    	b.gt	0x1000003a0 <__Z25test_nested_short_circuitiii+0x40>
10000038c: 14000001    	b	0x100000390 <__Z25test_nested_short_circuitiii+0x30>
100000390: b94003e8    	ldr	w8, [sp]
100000394: 71000108    	subs	w8, w8, #0x0
100000398: 540000ad    	b.le	0x1000003ac <__Z25test_nested_short_circuitiii+0x4c>
10000039c: 14000001    	b	0x1000003a0 <__Z25test_nested_short_circuitiii+0x40>
1000003a0: 52800028    	mov	w8, #0x1                ; =1
1000003a4: b9000fe8    	str	w8, [sp, #0xc]
1000003a8: 14000003    	b	0x1000003b4 <__Z25test_nested_short_circuitiii+0x54>
1000003ac: b9000fff    	str	wzr, [sp, #0xc]
1000003b0: 14000001    	b	0x1000003b4 <__Z25test_nested_short_circuitiii+0x54>
1000003b4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003b8: 910043ff    	add	sp, sp, #0x10
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: d10083ff    	sub	sp, sp, #0x20
1000003c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c8: 910043fd    	add	x29, sp, #0x10
1000003cc: 52800001    	mov	w1, #0x0                ; =0
1000003d0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d4: aa0103e0    	mov	x0, x1
1000003d8: 528000a2    	mov	w2, #0x5                ; =5
1000003dc: 97ffffe1    	bl	0x100000360 <__Z25test_nested_short_circuitiii>
1000003e0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e4: 910083ff    	add	sp, sp, #0x20
1000003e8: d65f03c0    	ret
