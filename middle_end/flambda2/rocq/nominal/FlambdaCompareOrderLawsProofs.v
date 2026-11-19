(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Proofs of the statements of FlambdaCompareOrderLawsSpec.v. There is
   no statement here, and nothing here needs review.

   The two proofs are those of FlambdaCompareAlphaProofs.v for
   [expr_eqb] ([expr_eqb_trans], [expr_eqb_renamed]), gone through
   again for the comparison that lets its second term state less.
   They part only at the metadata of two paired binders: there the
   transitivity is that of the order on what two binders state
   (FlambdaSubkindOrderProofs.params_say_at_least_trans), and a
   renaming changes no metadata. The lemmas of that file on what the
   two comparisons share (simples, primitives, applications, the maps
   of paired names) are used as they are. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaSubkindOrderSpec FlambdaSubkindOrderProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaCompareSpec FlambdaCompareProofs
  FlambdaCompareAlphaSpec FlambdaCompareAlphaProofs
  FlambdaCompareOrderSpec FlambdaCompareOrderProofs
  FlambdaCompareOrderLawsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Transitivity                                                             *)
(* ======================================================================== *)

Lemma handler_metadata_says_at_least_trans (m m' m'' : handler_metadata) :
  handler_metadata_says_at_least m m' = true ->
  handler_metadata_says_at_least m' m'' = true ->
  handler_metadata_says_at_least m m'' = true.
Proof.
  unfold handler_metadata_says_at_least. intros H1 H2.
  apply andb_true_iff in H1 as [H1 Hexn1].
  apply andb_true_iff in H1 as [Hkinds1 Hsay1].
  apply andb_true_iff in H2 as [H2 Hexn2].
  apply andb_true_iff in H2 as [Hkinds2 Hsay2].
  apply kinds_eqb_eq in Hkinds1, Hkinds2.
  apply Bool.eqb_prop in Hexn1, Hexn2.
  rewrite Hkinds1, Hkinds2, Hexn1, Hexn2, kinds_eqb_refl, Bool.eqb_reflx,
    (params_say_at_least_trans _ _ _ Hsay1 Hsay2).
  reflexivity.
Qed.

Lemma code_metadata_says_at_least_trans (m m' m'' : code_metadata) :
  code_metadata_says_at_least m m' = true ->
  code_metadata_says_at_least m' m'' = true ->
  code_metadata_says_at_least m m'' = true.
Proof.
  unfold code_metadata_says_at_least. intros H1 H2.
  apply andb_true_iff in H1 as [H1 Hstub1].
  apply andb_true_iff in H1 as [H1 Htupled1].
  apply andb_true_iff in H1 as [H1 Hresults1].
  apply andb_true_iff in H1 as [H1 Hresult1].
  apply andb_true_iff in H1 as [Hparams1 Hsay1].
  apply andb_true_iff in H2 as [H2 Hstub2].
  apply andb_true_iff in H2 as [H2 Htupled2].
  apply andb_true_iff in H2 as [H2 Hresults2].
  apply andb_true_iff in H2 as [H2 Hresult2].
  apply andb_true_iff in H2 as [Hparams2 Hsay2].
  apply arity_eqb_eq in Hparams1, Hparams2, Hresult1, Hresult2.
  apply Bool.eqb_prop in Hstub1, Hstub2, Htupled1, Htupled2.
  rewrite Hparams1, Hparams2, Hresult1, Hresult2, Hstub1, Hstub2, Htupled1,
    Htupled2, !arity_eqb_refl, !Bool.eqb_reflx,
    (params_say_at_least_trans _ _ _ Hsay1 Hsay2),
    (params_say_at_least_trans _ _ _ Hresults1 Hresults2).
  reflexivity.
Qed.

(* Under maps of paired names that compose. *)
Fixpoint expr_says_at_least_trans (e : expr) :
  ∀ vm1 vm2 vm km1 km2 km : name_map,
    composes vm1 vm2 vm ->
    composes km1 km2 km ->
    ∀ e' : expr,
      expr_says_at_least vm1 km1 e e' = true ->
      ∀ e'' : expr,
        expr_says_at_least vm2 km2 e' e'' = true ->
        expr_says_at_least vm km e e'' = true
with handler_says_at_least_trans (h : continuation_handler) :
  ∀ vm1 vm2 vm km1 km2 km : name_map,
    composes vm1 vm2 vm ->
    composes km1 km2 km ->
    ∀ h' : continuation_handler,
      handler_says_at_least vm1 km1 h h' = true ->
      ∀ h'' : continuation_handler,
        handler_says_at_least vm2 km2 h' h'' = true ->
        handler_says_at_least vm km h h'' = true
with binding_says_at_least_trans (binding : static_binding) :
  ∀ vm1 vm2 vm : name_map,
    composes vm1 vm2 vm ->
    ∀ binding' : static_binding,
      binding_says_at_least vm1 binding binding' = true ->
      ∀ binding'' : static_binding,
        binding_says_at_least vm2 binding' binding'' = true ->
        binding_says_at_least vm binding binding'' = true
with code_says_at_least_trans (c : code) :
  ∀ c' : code,
    code_says_at_least c c' = true ->
    ∀ c'' : code,
      code_says_at_least c' c'' = true -> code_says_at_least c c'' = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vm1 vm2 vm km1 km2 km Hvm Hkm
        [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
        | ap' | ac' | scrutinee' arms' | ];
      simpl; try discriminate; intros H1
        [ x'' kind'' d'' body'' | bindings'' body'' | k'' h'' body''
        | handlers'' body'' | ap'' | ac'' | scrutinee'' arms'' | ];
      simpl; try discriminate; intros H2.
    + (* Let *)
      apply andb_true_iff in H1 as [Hd1 Hbody1].
      apply andb_true_iff in Hd1 as [Hkind1 Hd1].
      apply andb_true_iff in H2 as [Hd2 Hbody2].
      apply andb_true_iff in Hd2 as [Hkind2 Hd2].
      apply kind_eqb_eq in Hkind1, Hkind2. subst kind' kind''.
      rewrite kind_eqb_refl, (named_eqb_trans _ _ _ Hvm _ _ _ Hd1 Hd2).
      exact (expr_says_at_least_trans body _ _ _ _ _ _
               (composes_cons _ _ _ x x' x'' Hvm) Hkm _ Hbody1 _ Hbody2).
    + (* Let_static *)
      apply andb_true_iff in H1 as [Hbindings1 Hbody1].
      apply andb_true_iff in H2 as [Hbindings2 Hbody2].
      rewrite (expr_says_at_least_trans body _ _ _ _ _ _ Hvm Hkm _ Hbody1 _
        Hbody2), andb_true_r.
      clear Hbody1 Hbody2. revert bindings' Hbindings1 bindings'' Hbindings2.
      induction bindings as [ | binding bindings IH];
        intros [ | binding' bindings'] H1 [ | binding'' bindings''] H2;
        simpl in *; try discriminate H1; try discriminate H2;
        [ reflexivity | ].
      apply andb_true_iff in H1 as [Hfirst1 Hrest1].
      apply andb_true_iff in H2 as [Hfirst2 Hrest2].
      rewrite (binding_says_at_least_trans binding _ _ _ Hvm _ Hfirst1 _
        Hfirst2), (IH _ Hrest1 _ Hrest2).
      reflexivity.
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in H1 as [Hh1 Hbody1].
      apply andb_true_iff in H2 as [Hh2 Hbody2].
      rewrite (handler_says_at_least_trans h _ _ _ _ _ _ Hvm Hkm _ Hh1 _ Hh2).
      exact (expr_says_at_least_trans body _ _ _ _ _ _ Hvm
               (composes_cons _ _ _ k k' k'' Hkm) _ Hbody1 _ Hbody2).
    + (* Let_cont_recursive *)
      revert H1 H2.
      destruct (combine_names (map fst handlers) (map fst handlers') km1)
        as [km1' | ] eqn:Hnames1; [ | discriminate ].
      destruct (combine_names (map fst handlers') (map fst handlers'') km2)
        as [km2' | ] eqn:Hnames2; [ | intros _; discriminate ].
      intros H1 H2.
      destruct (combine_names_composes _ _ _ _ _ _ _ _ Hkm Hnames1 Hnames2)
        as (km' & -> & Hkm').
      apply andb_true_iff in H1 as [Hhandlers1 Hbody1].
      apply andb_true_iff in H2 as [Hhandlers2 Hbody2].
      rewrite (expr_says_at_least_trans body _ _ _ _ _ _ Hvm Hkm' _ Hbody1 _
        Hbody2), andb_true_r.
      clear Hbody1 Hbody2 Hnames1 Hnames2.
      revert handlers' Hhandlers1 handlers'' Hhandlers2.
      induction handlers as [ | [k h] handlers IH];
        intros [ | [k' h'] handlers'] H1 [ | [k'' h''] handlers''] H2;
        simpl in *; try discriminate H1; try discriminate H2;
        [ reflexivity | ].
      apply andb_true_iff in H1 as [Hfirst1 Hrest1].
      apply andb_true_iff in H2 as [Hfirst2 Hrest2].
      rewrite (handler_says_at_least_trans h _ _ _ _ _ _ Hvm Hkm' _ Hfirst1
        _ Hfirst2), (IH _ Hrest1 _ Hrest2).
      reflexivity.
    + (* Apply *)
      exact (apply_eqb_trans _ _ _ Hvm _ _ _ Hkm _ _ _ H1 H2).
    + (* Apply_cont *)
      exact (apply_cont_eqb_trans _ _ _ Hvm _ _ _ Hkm _ _ _ H1 H2).
    + (* Switch *)
      apply andb_true_iff in H1 as [Hscrutinee1 Harms1].
      apply andb_true_iff in H2 as [Hscrutinee2 Harms2].
      rewrite (simple_eqb_trans _ _ _ Hvm _ _ _ Hscrutinee1 Hscrutinee2).
      refine (forall2b_trans _ _ _ _ _ _ _ Harms1 Harms2).
      intros arm arm' arm'' Harm1 Harm2.
      apply andb_true_iff in Harm1 as [Hz1 Hac1].
      apply andb_true_iff in Harm2 as [Hz2 Hac2].
      apply Z.eqb_eq in Hz1, Hz2. rewrite Hz1, Hz2, Z.eqb_refl.
      exact (apply_cont_eqb_trans _ _ _ Hvm _ _ _ Hkm _ _ _ Hac1 Hac2).
    + (* Invalid *)
      reflexivity.
  - destruct h as [params metadata handler].
    intros vm1 vm2 vm km1 km2 km Hvm Hkm [params' metadata' handler'] H1
      [params'' metadata'' handler''] H2.
    simpl in *. revert H1 H2.
    destruct (bind_names params params' vm1) as [vm1' | ] eqn:Hbind1;
      [ | discriminate ].
    destruct (bind_names params' params'' vm2) as [vm2' | ] eqn:Hbind2;
      [ | intros _; discriminate ].
    intros H1 H2.
    destruct (bind_names_composes params _ _ _ _ _ _ _ Hvm Hbind1 Hbind2)
      as (vm' & -> & Hvm').
    apply andb_true_iff in H1 as [Hmetadata1 H1].
    apply andb_true_iff in H2 as [Hmetadata2 H2].
    rewrite (handler_metadata_says_at_least_trans _ _ _ Hmetadata1
      Hmetadata2).
    exact (expr_says_at_least_trans handler _ _ _ _ _ _ Hvm' Hkm _ H1 _ H2).
  - destruct binding as [id c | id | sym const];
      intros vm1 vm2 vm Hvm [id' c' | id' | sym' const']; simpl;
      try discriminate; intros H1 [id'' c'' | id'' | sym'' const'']; simpl;
      try discriminate; intros H2.
    + apply andb_true_iff in H1 as [Hid1 Hc1].
      apply andb_true_iff in H2 as [Hid2 Hc2].
      apply Pos.eqb_eq in Hid1, Hid2. subst id' id''. rewrite Pos.eqb_refl.
      exact (code_says_at_least_trans c _ Hc1 _ Hc2).
    + apply Pos.eqb_eq in H1, H2. subst id' id''. apply Pos.eqb_refl.
    + apply andb_true_iff in H1 as [Hsym1 Hconst1].
      apply andb_true_iff in H2 as [Hsym2 Hconst2].
      apply Pos.eqb_eq in Hsym1, Hsym2. subst sym' sym''.
      rewrite Pos.eqb_refl.
      exact (static_const_eqb_trans _ _ _ Hvm _ _ _ Hconst1 Hconst2).
  - destruct c as [params my_closure ret exn metadata body].
    intros [params' my_closure' ret' exn' metadata' body'] H1
      [params'' my_closure'' ret'' exn'' metadata'' body''] H2.
    simpl in *. revert H1 H2.
    destruct (bind_names params params' [(my_closure, my_closure')])
      as [vm1 | ] eqn:Hbind1; [ | discriminate ].
    destruct (bind_names params' params'' [(my_closure', my_closure'')])
      as [vm2 | ] eqn:Hbind2; [ | intros _; discriminate ].
    intros H1 H2.
    destruct (bind_names_composes params _ _ _ _
                [(my_closure, my_closure'')] _ _
                (composes_cons _ _ _ my_closure my_closure' my_closure''
                   composes_nil)
                Hbind1 Hbind2)
      as (vm & -> & Hvm).
    apply andb_true_iff in H1 as [Hmetadata1 H1].
    apply andb_true_iff in H2 as [Hmetadata2 H2].
    rewrite (code_metadata_says_at_least_trans _ _ _ Hmetadata1 Hmetadata2).
    exact (expr_says_at_least_trans body _ _ _ _ _ _ Hvm
             (composes_cons _ _ _ ret ret' ret''
                (composes_cons _ _ _ exn exn' exn'' composes_nil))
             _ H1 _ H2).
Qed.

Theorem expr_says_at_least_transitive :
  expr_says_at_least_transitive_statement.
Proof.
  intros e e' e'' H1 H2.
  exact (expr_says_at_least_trans e _ _ _ _ _ _ composes_nil composes_nil _
           H1 _ H2).
Qed.

(* ======================================================================== *)
(* The comparison does not depend on the names                              *)
(* ======================================================================== *)

Section Comparison_renamed.

Variable r : renaming.

Fixpoint expr_says_at_least_renamed (e : expr) {struct e} :
  ∀ (vm km : name_map) (e' : expr),
    expr_says_at_least (rename_name_map (rename_variable r) vm)
      (rename_name_map (rename_continuation r) km) (rename_expr r e)
      (rename_expr r e')
    = expr_says_at_least vm km e e'
with handler_says_at_least_renamed (h : continuation_handler) {struct h} :
  ∀ (vm km : name_map) (h' : continuation_handler),
    handler_says_at_least (rename_name_map (rename_variable r) vm)
      (rename_name_map (rename_continuation r) km) (rename_handler r h)
      (rename_handler r h')
    = handler_says_at_least vm km h h'
with binding_says_at_least_renamed (binding : static_binding) {struct binding} :
  ∀ (vm : name_map) (binding' : static_binding),
    binding_says_at_least (rename_name_map (rename_variable r) vm)
      (rename_static_binding r binding) (rename_static_binding r binding')
    = binding_says_at_least vm binding binding'
with code_says_at_least_renamed (c : code) {struct c} :
  ∀ c' : code,
    code_says_at_least (rename_code r c) (rename_code r c')
    = code_says_at_least c c'.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vm km
        [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
        | ap' | ac' | scrutinee' arms' | ];
      simpl; try reflexivity.
    + (* Let *)
      rewrite named_eqb_equivariant. f_equal.
      exact (expr_says_at_least_renamed body ((x, x') :: vm) km body').
    + (* Let_static *)
      rewrite (expr_says_at_least_renamed body vm km body'). f_equal.
      revert bindings'.
      induction bindings as [ | binding bindings IH];
        intros [ | binding' bindings']; simpl; try reflexivity.
      rewrite (binding_says_at_least_renamed binding vm binding'), IH.
      reflexivity.
    + (* Let_cont_non_recursive *)
      rewrite (handler_says_at_least_renamed h vm km h'). f_equal.
      exact (expr_says_at_least_renamed body vm ((k, k') :: km) body').
    + (* Let_cont_recursive *)
      fold (rename_handlers r handlers). fold (rename_handlers r handlers').
      rewrite !map_fst_rename_handlers.
      rewrite (combine_names_renamed (rename_continuation r)).
      destruct (combine_names (map fst handlers) (map fst handlers') km)
        as [km' | ]; simpl; [ | reflexivity ].
      rewrite (expr_says_at_least_renamed body vm km' body'). f_equal.
      unfold rename_handlers. revert handlers'.
      induction handlers as [ | [k0 h] handlers IH];
        intros [ | [k0' h'] handlers']; simpl; try reflexivity.
      rewrite (handler_says_at_least_renamed h vm km' h'), IH. reflexivity.
    + (* Apply *)
      apply apply_eqb_equivariant.
    + (* Apply_cont *)
      apply apply_cont_eqb_equivariant.
    + (* Switch *)
      rewrite simple_eqb_equivariant. f_equal.
      rewrite forall2b_map_both. apply forall2b_ext. intros arm arm'.
      simpl. rewrite apply_cont_eqb_equivariant. reflexivity.
  - destruct h as [params metadata handler].
    intros vm km [params' metadata' handler']. simpl.
    rewrite (bind_names_renamed (rename_variable r)).
    destruct (bind_names params params' vm) as [vm' | ]; simpl;
      [ | reflexivity ].
    f_equal. exact (expr_says_at_least_renamed handler vm' km handler').
  - destruct binding as [id c | id | sym const];
      intros vm [id' c' | id' | sym' const']; simpl; try reflexivity.
    + rewrite (code_says_at_least_renamed c c'). reflexivity.
    + rewrite static_const_eqb_equivariant. reflexivity.
  - destruct c as [params my_closure ret exn metadata body].
    intros [params' my_closure' ret' exn' metadata' body']. simpl.
    change [(rename_variable r my_closure, rename_variable r my_closure')]
      with (rename_name_map (rename_variable r) [(my_closure, my_closure')]).
    rewrite (bind_names_renamed (rename_variable r)).
    destruct (bind_names params params' [(my_closure, my_closure')])
      as [vm' | ]; simpl; [ | reflexivity ].
    f_equal.
    exact (expr_says_at_least_renamed body vm' [(ret, ret'); (exn, exn')]
             body').
Qed.

End Comparison_renamed.

Theorem expr_says_at_least_equivariant :
  expr_says_at_least_equivariant_statement.
Proof.
  intros r vm km e e'. apply expr_says_at_least_renamed.
Qed.
