
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf060-switch-large-values/cf060-switch-large-values_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_switch_largei>:
100000360: 7104ac1f    	cmp	w0, #0x12b
100000364: 540000ec    	b.gt	0x100000380 <__Z17test_switch_largei+0x20>
100000368: 7101901f    	cmp	w0, #0x64
10000036c: 540001a0    	b.eq	0x1000003a0 <__Z17test_switch_largei+0x40>
100000370: 7103201f    	cmp	w0, #0xc8
100000374: 54000121    	b.ne	0x100000398 <__Z17test_switch_largei+0x38>
100000378: 52800040    	mov	w0, #0x2                ; =2
10000037c: d65f03c0    	ret
100000380: 7104b01f    	cmp	w0, #0x12c
100000384: 54000120    	b.eq	0x1000003a8 <__Z17test_switch_largei+0x48>
100000388: 710fa01f    	cmp	w0, #0x3e8
10000038c: 54000061    	b.ne	0x100000398 <__Z17test_switch_largei+0x38>
100000390: 52800080    	mov	w0, #0x4                ; =4
100000394: d65f03c0    	ret
100000398: 52800000    	mov	w0, #0x0                ; =0
10000039c: d65f03c0    	ret
1000003a0: 52800020    	mov	w0, #0x1                ; =1
1000003a4: d65f03c0    	ret
1000003a8: 52800060    	mov	w0, #0x3                ; =3
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: 52800060    	mov	w0, #0x3                ; =3
1000003b4: d65f03c0    	ret
