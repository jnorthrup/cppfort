
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf014-ternary-function-calls/cf014-ternary-function-calls_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z13positive_funcv>:
100000360: 52800020    	mov	w0, #0x1                ; =1
100000364: d65f03c0    	ret

0000000100000368 <__Z13negative_funcv>:
100000368: 12800000    	mov	w0, #-0x1               ; =-1
10000036c: d65f03c0    	ret

0000000100000370 <__Z18test_ternary_callsi>:
100000370: d10083ff    	sub	sp, sp, #0x20
100000374: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000378: 910043fd    	add	x29, sp, #0x10
10000037c: b81fc3a0    	stur	w0, [x29, #-0x4]
100000380: b85fc3a8    	ldur	w8, [x29, #-0x4]
100000384: 71000108    	subs	w8, w8, #0x0
100000388: 540000ad    	b.le	0x10000039c <__Z18test_ternary_callsi+0x2c>
10000038c: 14000001    	b	0x100000390 <__Z18test_ternary_callsi+0x20>
100000390: 97fffff4    	bl	0x100000360 <__Z13positive_funcv>
100000394: b9000be0    	str	w0, [sp, #0x8]
100000398: 14000004    	b	0x1000003a8 <__Z18test_ternary_callsi+0x38>
10000039c: 97fffff3    	bl	0x100000368 <__Z13negative_funcv>
1000003a0: b9000be0    	str	w0, [sp, #0x8]
1000003a4: 14000001    	b	0x1000003a8 <__Z18test_ternary_callsi+0x38>
1000003a8: b9400be0    	ldr	w0, [sp, #0x8]
1000003ac: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b0: 910083ff    	add	sp, sp, #0x20
1000003b4: d65f03c0    	ret

00000001000003b8 <_main>:
1000003b8: d10083ff    	sub	sp, sp, #0x20
1000003bc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c0: 910043fd    	add	x29, sp, #0x10
1000003c4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c8: 528000a0    	mov	w0, #0x5                ; =5
1000003cc: 97ffffe9    	bl	0x100000370 <__Z18test_ternary_callsi>
1000003d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d4: 910083ff    	add	sp, sp, #0x20
1000003d8: d65f03c0    	ret
