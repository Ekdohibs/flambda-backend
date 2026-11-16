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



(* Proofs of the statements of FlambdaReifySpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaValueProofs
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaTypeWellFormedSpec
  FlambdaTypeWellFormedProofs FlambdaTypeAccessSpec FlambdaTypeAccessProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaExpandHead FlambdaProvers FlambdaProversSpec
  FlambdaProversProofs FlambdaReify FlambdaReifySpec.
Import ListNotations.

Theorem unboxed_number_kind : unboxed_number_kind_statement.
Proof. intros k. destruct k; reflexivity. Qed.

(* ======================================================================== *)
(* The fields of a block                                                    *)
(* ======================================================================== *)

(* The [Simple] that [Reify.reify_field] answers for a field has the
   value of the field. *)
Lemma reify_field_sound (statics : static_env) (valuation : env)
    (tenv : Typing_env.t)
    (var_allowed : Alloc_mode.t -> variable -> option bool)
    (alloc_mode : Alloc_mode.t) (field : flambda_type * flambda_kind)
    (s : simple) (w : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  Reify.reify_field tenv var_allowed alloc_mode field = Some (Some s) ->
  type_satisfied statics valuation w (fst field) ->
  eval_simple valuation s = Some w.
Proof.
  intros Hsat Hfield Hw. unfold Reify.reify_field in Hfield.
  destruct
    (Provers.prove_equals_to_simple_of_kind tenv (fst field) (snd field))
    as [[s' | ] | ] eqn:Hproved;
    try discriminate Hfield.
  pose proof
    (prove_equals_to_simple_of_kind_sound_proof (snd field) statics valuation
       tenv (fst field) s' w Hsat I Hw Hproved)
    as Hs.
  destruct s' as [x | sym | c]; simpl in Hfield.
  - destruct (var_allowed alloc_mode x) as [[ | ] | ];
      try discriminate Hfield.
    injection Hfield as <-. exact Hs.
  - injection Hfield as <-. exact Hs.
  - injection Hfield as <-. exact Hs.
Qed.

Lemma reify_fields_length (tenv : Typing_env.t)
    (var_allowed : Alloc_mode.t -> variable -> option bool)
    (alloc_mode : Alloc_mode.t)
    (fields : list (flambda_type * flambda_kind)) :
  ∀ simples : list simple,
    Reify.reify_fields tenv var_allowed alloc_mode fields = Some simples ->
    length simples <= length fields.
Proof.
  induction fields as [ | field fields IH]; intros simples Hfields;
    simpl in Hfields.
  - injection Hfields as <-. simpl. lia.
  - destruct (Reify.reify_field tenv var_allowed alloc_mode field)
      as [simple | ];
      [ | discriminate Hfields ].
    destruct (Reify.reify_fields tenv var_allowed alloc_mode fields)
      as [simples' | ];
      [ | discriminate Hfields ].
    specialize (IH simples' eq_refl). simpl in Hfields.
    destruct simple as [s | ]; injection Hfields as <-; simpl; lia.
Qed.

(* If [Reify.reify_fields] answers as many [Simple]s as there are
   fields, each has the value of its field. *)
Lemma reify_fields_sound (statics : static_env) (valuation : env)
    (tenv : Typing_env.t)
    (var_allowed : Alloc_mode.t -> variable -> option bool)
    (alloc_mode : Alloc_mode.t)
    (fields : list (flambda_type * flambda_kind)) :
  typing_env_satisfied_strictly statics valuation tenv ->
  ∀ simples : list simple,
    Reify.reify_fields tenv var_allowed alloc_mode fields = Some simples ->
    length fields = length simples ->
    Forall2
      (fun field s =>
         ∀ w : value,
           type_satisfied statics valuation w (fst field) ->
           eval_simple valuation s = Some w)
      fields simples.
Proof.
  intros Hsat.
  induction fields as [ | field fields IH]; intros simples Hfields Hlength;
    simpl in Hfields.
  - injection Hfields as <-. constructor.
  - destruct (Reify.reify_field tenv var_allowed alloc_mode field)
      as [simple | ] eqn:Hfield;
      [ | discriminate Hfields ].
    destruct (Reify.reify_fields tenv var_allowed alloc_mode fields)
      as [simples' | ] eqn:Hrest;
      [ | discriminate Hfields ].
    pose proof (reify_fields_length _ _ _ _ _ Hrest) as Hle.
    simpl in Hfields.
    destruct simple as [s | ]; injection Hfields as <-; simpl in Hlength.
    + constructor.
      * intros w Hw.
        exact (reify_field_sound _ _ _ _ _ _ _ _ Hsat Hfield Hw).
      * apply IH; [ reflexivity | lia ].
    + lia.
Qed.

Lemma forall2_combine_fst {A B C : Type} (Q : A -> C -> Prop) (l : list A) :
  ∀ (l' : list B) (l'' : list C),
    length l' = length l ->
    Forall2 (fun p c => Q (fst p) c) (combine l l') l'' ->
    Forall2 Q l l''.
Proof.
  induction l as [ | a l IH]; intros [ | b l'] l'' Hlength Hall;
    simpl in Hlength, Hall; try discriminate Hlength.
  - inversion Hall. constructor.
  - inversion Hall as [ | p c ps cs Hp Hrest]; subst.
    constructor; [ exact Hp | ].
    apply (IH l' cs); [ lia | exact Hrest ].
Qed.

Lemma forall2_fields {A : Type} (P : value -> flambda_type -> Prop)
    (Q : flambda_type -> A -> Prop) (R : A -> value -> Prop)
    (fields : list value) (tys : list flambda_type) :
  (∀ (field : value) (ty : flambda_type) (a : A),
    P field ty -> Q ty a -> R a field) ->
  Forall2 P fields tys ->
  ∀ l : list A, Forall2 Q tys l -> Forall2 R l fields.
Proof.
  intros Hcompose. induction 1 as [ | field ty fields tys Hfield Hall IH];
    intros l Hq; inversion Hq; subst; constructor; eauto.
Qed.

(* ======================================================================== *)
(* Reification                                                              *)
(* ======================================================================== *)

Lemma reify_expanded_sound (statics : static_env) (valuation : env)
    (tenv : Typing_env.t)
    (var_allowed : Alloc_mode.t -> variable -> option bool)
    (try_canonical_simple result : Reification_result.t) (t : flambda_type)
    (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  well_formed_input tenv t ->
  type_satisfied statics valuation v t ->
  reification_holds statics valuation try_canonical_simple v ->
  Reify.reify_expanded var_allowed tenv try_canonical_simple t
  = Some result ->
  reification_holds statics valuation result v.
Proof.
  intros Hsat Hin Hv Htry. unfold Reify.reify_expanded.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | discriminate ].
  simpl.
  pose proof (provers_expand_head_well_formed _ _ _ Hin Hd) as Hwf.
  destruct d as [oub | oub | k oub | oub | oub].
  - (* Kind value. *)
    pose proof (expanded_value_cases _ _ _ _ _ _ Hsat Hv Hd)
      as Hcases.
    destruct oub as [ | [non_null is_null] | ];
      [ intros [= <-]; exact Htry | | destruct Hcases ].
    destruct is_null as [ | x];
      [ | destruct non_null as [ | [ | | | | ] | ]; intros [= <-];
          exact Htry ].
    destruct non_null as [ | head | ];
      [ intros [= <-]; exact Htry | | intros [= <-]; exact Htry ].
    destruct Hcases as [[_ Hnot] | [_ Hhead]]; [ contradiction | ].
    assert (Hwf_head : well_formed_head tenv head).
    { exact (well_formed_input_to_head _ _ _ _ Hin Hd). }
    destruct head as [is_int immediates get_tag blocks extensions is_unique
                     | alloc_mode | k contents alloc_mode
                     | by_function_slot alloc_mode | strs];
      try (intros [= <-]; exact Htry).
    + (* A variant. *)
      destruct blocks as [blocks | ]; [ | intros [= <-]; exact Htry ].
      destruct immediates as [imms | ]; [ | intros [= <-]; exact Htry ].
      destruct (Provers.is_bottom tenv imms) as [[ | ] | ] eqn:Himms;
        simpl; [ | | discriminate ].
      * (* No immediate: a block. *)
        destruct
          (variant_no_immediates _ _ _ _ _ _ _ _ _ _ Hsat
             Hhead Himms)
          as [_ [tag' [fields [Hblock Hblocks]]]].
        destruct (row_like_for_blocks_get_singleton blocks)
          as [[[[[tag shape] size] tys] alloc_mode] | ] eqn:Hsingleton;
          [ | intros [= <-]; exact Htry ].
        destruct
          (get_singleton_satisfied _ _ _ _ _ _ _ _ _ _ Hsingleton Hblocks)
          as [-> [Hlength [Hshape Hfields]]].
        destruct (Nat.eqb size (length tys)) eqn:Hsize; simpl;
          [ | discriminate ].
        apply Nat.eqb_eq in Hsize.
        destruct (Nat.eqb (length (field_kinds shape (length tys)))
                    (length tys))
          eqn:Hkinds;
          simpl; [ | discriminate ].
        apply Nat.eqb_eq in Hkinds.
        destruct shape as [shape | ]; [ | intros [= <-]; exact Htry ].
        destruct (Provers.tag_is_scannable tag); cbn [negb];
          [ | discriminate ].
        destruct
          (Reify.try_to_reify_fields tenv var_allowed alloc_mode
             (combine tys (field_kinds (Scannable shape) (length tys))))
          as [[simples | ] | ] eqn:Htry_fields;
          cbv beta iota; [ | intros [= <-]; exact Htry | discriminate ].
        intros [= <-]. exists fields.
        split; [ exact Hblock | ]. split; [ exact Hshape | ].
        unfold Reify.try_to_reify_fields in Htry_fields.
        destruct
          (Reify.reify_fields tenv var_allowed alloc_mode
             (combine tys (field_kinds (Scannable shape) (length tys))))
          as [simples' | ] eqn:Hreified;
          [ | discriminate Htry_fields ].
        cbv beta iota in Htry_fields.
        destruct
          (Nat.eqb
             (length
                (combine tys (field_kinds (Scannable shape) (length tys))))
             (length simples'))
          eqn:Hall;
          [ | discriminate Htry_fields ].
        injection Htry_fields as <-. apply Nat.eqb_eq in Hall.
        pose proof (reify_fields_sound _ _ _ _ _ _ Hsat _ Hreified Hall)
          as Hsimples.
        apply
          (forall2_combine_fst
             (fun ty s =>
                ∀ w : value,
                  type_satisfied statics valuation w ty ->
                  eval_simple valuation s = Some w)
             _ _ _ Hkinds)
          in Hsimples.
        apply eval_simples_of_fields.
        assert (Hlength' : length fields = length tys).
        { rewrite Hlength. exact Hsize. }
        apply (fields_satisfied_Forall2 _ _ _ Hlength') in Hfields.
        refine (forall2_fields _ _ _ _ _ _ Hfields simples' Hsimples).
        intros field ty s Hfield Hs. exact (Hs field Hfield).
      * (* An immediate may be there. *)
        destruct (row_like_for_blocks_is_bottom blocks) eqn:Hbottom;
          [ | intros [= <-]; exact Htry ].
        destruct (variant_no_blocks _ _ _ _ _ _ _ _ _ Hhead Hbottom)
          as [i [-> Hi]].
        pose proof
          (meet_naked_immediates_sound_proof statics
             valuation tenv imms (Value_const (Naked_immediate i)) Hsat
             (well_formed_head_immediates _ _ _ _ _ _ _ Hwf_head) Hi
             eq_refl)
          as Hmeet.
        destruct (Provers.meet_naked_immediates tenv imms)
          as [[numbers | | ] | ];
          simpl; [ | intros [= <-]; exact Htry | destruct Hmeet
                 | discriminate ].
        destruct Hmeet as [i' [[= <-] Hin_numbers]].
        destruct (Keyed.get_singleton Z.eqb numbers) as [imm | ]
          eqn:Hsingleton;
          intros [= <-]; [ | exact Htry ].
        simpl. rewrite (get_singleton_in _ _ _ Hsingleton Hin_numbers).
        reflexivity.
    + (* A boxed number. *)
      rewrite boxed_number_satisfied in Hhead.
      destruct Hhead as [c [Hboxed [Hkind Hc]]].
      assert (Hkind' :
        kind_of_value (Value_const c)
        = Naked_number (Reify.unboxed_number_kind k)).
      { rewrite unboxed_number_kind. exact Hkind. }
      assert (Hnot_immediate :
        Reify.unboxed_number_kind k <> Kind_naked_immediate).
      { destruct k; discriminate. }
      pose proof
        (meet_naked_number_sound_proof
           (Reify.unboxed_number_kind k) Hnot_immediate statics valuation
           tenv contents (Value_const c) Hsat I Hc Hkind')
        as Hmeet.
      destruct
        (Provers.meet_naked_number (Reify.unboxed_number_kind k) tenv
           contents)
        as [[numbers | | ] | ];
        simpl; [ | intros [= <-]; exact Htry | destruct Hmeet
               | discriminate ].
      destruct Hmeet as [c' [n [[= <-] [Hkind_c [Hnumber Hin_numbers]]]]].
      destruct (Keyed.get_singleton Z.eqb numbers) as [n' | ] eqn:Hsingleton;
        intros [= <-]; [ | exact Htry ].
      simpl. rewrite Hboxed.
      rewrite (const_of_its_number _ _ _ Hkind_c Hnumber).
      rewrite (get_singleton_in _ _ _ Hsingleton Hin_numbers).
      reflexivity.
  - (* Kind naked immediate. *)
    pose proof
      (expanded_naked_immediate_cases _ _ _ _ _ _ Hsat Hv
         Hd)
      as Hcases.
    destruct oub as [ | head | ];
      [ intros [= <-]; exact Htry | | destruct Hcases ].
    pose proof (expanded_kind _ _ _ _ _ _ Hsat Hv Hd)
      as Hkind.
    pose proof (expand_head_descr_sound _ _ _ _ _ _ Hsat Hv
                  Hd)
      as Hexpanded.
    simpl in Hkind, Hexpanded.
    pose proof
      (meet_naked_immediates_sound_proof statics valuation
         tenv (Type_naked_immediate (Ok (No_alias head))) v Hsat
         (conj (proj1 Hin) Hwf) Hexpanded Hkind)
      as Hmeet.
    destruct
      (Provers.meet_naked_immediates tenv
         (Type_naked_immediate (Ok (No_alias head))))
      as [[numbers | | ] | ];
      simpl; [ | intros [= <-]; exact Htry | destruct Hmeet | discriminate ].
    destruct Hmeet as [i [-> Hin_numbers]].
    destruct (Keyed.get_singleton Z.eqb numbers) as [i' | ] eqn:Hsingleton;
      intros [= <-]; [ | exact Htry ].
    simpl. rewrite (get_singleton_in _ _ _ Hsingleton Hin_numbers).
    reflexivity.
  - (* A kind of naked numbers. *)
    pose proof
      (expanded_naked_number_cases _ _ _ _ _ _ _ Hsat Hv
         Hd)
      as Hcases.
    destruct oub as [ | numbers | ];
      [ intros [= <-]; exact Htry | | destruct Hcases ].
    destruct Hcases as [c [n [-> [Hkind_c [Hnumber Hin_numbers]]]]].
    destruct (Keyed.get_singleton Z.eqb numbers) as [n' | ] eqn:Hsingleton;
      intros [= <-]; [ | exact Htry ].
    simpl. rewrite (const_of_its_number _ _ _ Hkind_c Hnumber).
    rewrite (get_singleton_in _ _ _ Hsingleton Hin_numbers).
    reflexivity.
  - (* Rec_info. *)
    pose proof (expand_head_descr_sound _ _ _ _ _ _ Hsat Hv
                  Hd)
      as Hexpanded.
    destruct oub as [ | head | ]; intros [= <-]; try exact Htry.
    simpl in Hexpanded.
    exact (bottom_satisfied statics valuation v Rec_info Hexpanded).
  - (* Region. *)
    pose proof (expand_head_descr_sound _ _ _ _ _ _ Hsat Hv
                  Hd)
      as Hexpanded.
    destruct oub as [ | head | ]; intros [= <-]; try exact Htry.
    simpl in Hexpanded.
    exact (bottom_satisfied statics valuation v Region Hexpanded).
Qed.

Lemma reify_sound_proof
    (allowed_if_free_vars_defined_in : Typing_env.t)
    (var_is_defined_at_toplevel var_is_symbol_projection : variable -> bool)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (result : Reification_result.t) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  well_formed_input tenv t ->
  type_satisfied statics valuation v t ->
  Reify.reify allowed_if_free_vars_defined_in var_is_defined_at_toplevel
    var_is_symbol_projection tenv t
  = Some result ->
  reification_holds statics valuation result v.
Proof.
  intros Hsat Hin Hv. unfold Reify.reify.
  destruct
    (Typing_env.get_alias_then_canonical_simple_exn tenv
       (Some Name_mode.normal) None t)
    as [[s | ] | ] eqn:Hcanonical;
    [ | | discriminate ].
  - pose proof
      (get_alias_then_canonical_simple_satisfied (fun _ => False) _ _ _ _ _
         _ _ _ Hsat Hcanonical Hv)
      as Hs.
    destruct s as [x | sym | c]; simpl.
    + exact
        (reify_expanded_sound _ _ _ _ (Reification_result.Simple (Var x)) _
           _ _ Hsat Hin Hv Hs).
    + intros [= <-]. exact Hs.
    + intros [= <-]. exact Hs.
  - simpl.
    exact
      (reify_expanded_sound _ _ _ _ Reification_result.Cannot_reify _ _ _
         Hsat Hin Hv I).
Qed.

Theorem reify_sound : reify_sound_statement.
Proof. exact reify_sound_proof. Qed.

Theorem example_reify : example_reify_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.
