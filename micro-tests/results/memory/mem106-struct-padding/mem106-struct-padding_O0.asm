
/Users/jim/work/cppfort/micro-tests/results/memory/mem106-struct-padding/mem106-struct-padding_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_struct_paddingv>:
100000360: 52800180    	mov	w0, #0xc                ; =12
100000364: d65f03c0    	ret

0000000100000368 <_main>:
100000368: d10083ff    	sub	sp, sp, #0x20
10000036c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000370: 910043fd    	add	x29, sp, #0x10
100000374: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000378: 97fffffa    	bl	0x100000360 <__Z19test_struct_paddingv>
10000037c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000380: 910083ff    	add	sp, sp, #0x20
100000384: d65f03c0    	ret
