
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf009-if-side-effect/cf009-if-side-effect_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_if_side_effectiRi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: f90003e1    	str	x1, [sp]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: 71000108    	subs	w8, w8, #0x0
100000374: 540000ed    	b.le	0x100000390 <__Z19test_if_side_effectiRi+0x30>
100000378: 14000001    	b	0x10000037c <__Z19test_if_side_effectiRi+0x1c>
10000037c: b9400fe8    	ldr	w8, [sp, #0xc]
100000380: 531f7908    	lsl	w8, w8, #1
100000384: f94003e9    	ldr	x9, [sp]
100000388: b9000128    	str	w8, [x9]
10000038c: 14000001    	b	0x100000390 <__Z19test_if_side_effectiRi+0x30>
100000390: 910043ff    	add	sp, sp, #0x10
100000394: d65f03c0    	ret

0000000100000398 <_main>:
100000398: d10083ff    	sub	sp, sp, #0x20
10000039c: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a0: 910043fd    	add	x29, sp, #0x10
1000003a4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a8: 910023e1    	add	x1, sp, #0x8
1000003ac: b9000bff    	str	wzr, [sp, #0x8]
1000003b0: 528000a0    	mov	w0, #0x5                ; =5
1000003b4: 97ffffeb    	bl	0x100000360 <__Z19test_if_side_effectiRi>
1000003b8: b9400be0    	ldr	w0, [sp, #0x8]
1000003bc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c0: 910083ff    	add	sp, sp, #0x20
1000003c4: d65f03c0    	ret
