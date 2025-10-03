
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf042-do-while-execute-once/cf042-do-while-execute-once_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_do_while_oncev>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: 14000001    	b	0x10000036c <__Z18test_do_while_oncev+0xc>
10000036c: 52800548    	mov	w8, #0x2a               ; =42
100000370: b9000fe8    	str	w8, [sp, #0xc]
100000374: 14000001    	b	0x100000378 <__Z18test_do_while_oncev+0x18>
100000378: b9400fe0    	ldr	w0, [sp, #0xc]
10000037c: 910043ff    	add	sp, sp, #0x10
100000380: d65f03c0    	ret

0000000100000384 <_main>:
100000384: d10083ff    	sub	sp, sp, #0x20
100000388: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000038c: 910043fd    	add	x29, sp, #0x10
100000390: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000394: 97fffff3    	bl	0x100000360 <__Z18test_do_while_oncev>
100000398: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000039c: 910083ff    	add	sp, sp, #0x20
1000003a0: d65f03c0    	ret
