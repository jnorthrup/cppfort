
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf059-switch-empty-case/cf059-switch-empty-case_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z17test_switch_emptyi>:
1000003b0: 51000408    	sub	w8, w0, #0x1
1000003b4: 7100111f    	cmp	w8, #0x4
1000003b8: 540000a8    	b.hi	0x1000003cc <__Z17test_switch_emptyi+0x1c>
1000003bc: 90000009    	adrp	x9, 0x100000000
1000003c0: 910f7129    	add	x9, x9, #0x3dc
1000003c4: b8685920    	ldr	w0, [x9, w8, uxtw #2]
1000003c8: d65f03c0    	ret
1000003cc: 52800000    	mov	w0, #0x0                ; =0
1000003d0: d65f03c0    	ret

00000001000003d4 <_main>:
1000003d4: 52800f60    	mov	w0, #0x7b               ; =123
1000003d8: d65f03c0    	ret
