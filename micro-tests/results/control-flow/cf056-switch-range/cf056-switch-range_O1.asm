
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf056-switch-range/cf056-switch-range_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z17test_switch_rangei>:
1000003b0: 71004c1f    	cmp	w0, #0x13
1000003b4: 540000a8    	b.hi	0x1000003c8 <__Z17test_switch_rangei+0x18>
1000003b8: 90000008    	adrp	x8, 0x100000000
1000003bc: 910f6108    	add	x8, x8, #0x3d8
1000003c0: b8605900    	ldr	w0, [x8, w0, uxtw #2]
1000003c4: d65f03c0    	ret
1000003c8: 52800000    	mov	w0, #0x0                ; =0
1000003cc: d65f03c0    	ret

00000001000003d0 <_main>:
1000003d0: 52800040    	mov	w0, #0x2                ; =2
1000003d4: d65f03c0    	ret
