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


(* Proofs of the statements of FlambdaTypeAliasKindsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeAliasKindsSpec.
Import ListNotations.

Theorem aliases_with_kinds_unknown : aliases_with_kinds_unknown_statement.
Proof. intros [ | [] | | ]; reflexivity. Qed.

Theorem aliases_with_kinds_bottom : aliases_with_kinds_bottom_statement.
Proof. intros [ | [] | | ]; reflexivity. Qed.

Theorem aliases_with_kinds_alias : aliases_with_kinds_alias_statement.
Proof. intros [ | [] | | ] s; reflexivity. Qed.

Theorem aliases_with_kinds_get_alias
  : aliases_with_kinds_get_alias_statement.
Proof.
  intros [ty | ty | k ty | ty | ty] s; simpl;
    destruct ty as [ | [head | s'] | ]; try discriminate;
    intros [= ->]; reflexivity.
Qed.

Theorem aliases_with_kinds_type_for_const
  : aliases_with_kinds_type_for_const_statement.
Proof.
  intros c s k. destruct c; simpl;
    try (intros [[= <- <-] | []]; split; reflexivity);
    try (intros []).
  rewrite aliases_with_kinds_unknown. intros [].
Qed.

Theorem aliases_with_kinds_these_naked_immediates
  : aliases_with_kinds_these_naked_immediates_statement.
Proof.
  intros [ | i [ | i' imms]] s k; simpl.
  - intros [].
  - intros [[= <- <-] | []]. exists i. repeat split.
  - intros [].
Qed.

Theorem aliases_with_kinds_these_tagged_immediates
  : aliases_with_kinds_these_tagged_immediates_statement.
Proof.
  intros [ | i [ | i' imms]] s k; simpl.
  - intros [].
  - intros [[= <- <-] | []]. exists i. repeat split.
  - intros [].
Qed.

Lemma aliases_of_unknown_cases (tags : list Z) :
  flat_map
    (fun case : Z * Or_unknown.t row_like_block_case =>
       match snd case with
       | Or_unknown.Known case =>
         aliases_of_block_case aliases_with_kinds aliases_of_env_extension
           case
       | Or_unknown.Unknown => []
       end)
    (map (fun tag => (tag, Or_unknown.Unknown)) tags)
  = [].
Proof.
  induction tags as [ | tag tags IH]; simpl; [ reflexivity | exact IH ].
Qed.

Theorem aliases_with_kinds_none : aliases_with_kinds_none_statement.
Proof.
  repeat split; try reflexivity.
  intros tags alloc_mode. unfold blocks_with_these_tags. simpl.
  rewrite aliases_of_unknown_cases. reflexivity.
Qed.

Theorem aliases_with_kinds_create_variant
  : aliases_with_kinds_create_variant_statement.
Proof.
  intros is_unique immediates blocks extensions. simpl.
  rewrite app_nil_r. reflexivity.
Qed.

Theorem aliases_with_kinds_tag_immediate
  : aliases_with_kinds_tag_immediate_statement.
Proof. intros t. simpl. rewrite !app_nil_r. reflexivity. Qed.

Theorem aliases_with_kinds_box_number
  : aliases_with_kinds_box_number_statement.
Proof. intros k contents alloc_mode. simpl. apply app_nil_r. Qed.

Theorem aliases_with_kinds_immutable_block
  : aliases_with_kinds_immutable_block_statement.
Proof.
  intros is_unique tag shape alloc_mode fields. simpl.
  unfold aliases_of_block_case. simpl. rewrite !app_nil_r. reflexivity.
Qed.

Theorem aliases_with_kinds_create_closures
  : aliases_with_kinds_create_closures_statement.
Proof. intros alloc_mode by_function_slot. simpl. apply app_nil_r. Qed.

Theorem aliases_with_kinds_this_closure
  : aliases_with_kinds_this_closure_statement.
Proof.
  intros slot function_types closure_types value_slot_types alloc_mode.
  split; simpl; rewrite !app_nil_r; reflexivity.
Qed.

(* ======================================================================== *)
(* The names of equations, with their kinds                                 *)
(* ======================================================================== *)

Theorem equation_names_none : equation_names_none_statement.
Proof.
  split; [ intros [ | [] | | ]; reflexivity | ].
  split; [ intros [ | [] | | ]; reflexivity | ].
  split; [ intros [ | [] | | ] s; reflexivity | ].
  intros c. destruct c as [ | | | | | | | | | | | | | | k];
    try reflexivity.
  destruct k as [ | [] | | ]; reflexivity.
Qed.

Theorem equation_names_of_equations
  : equation_names_of_equations_statement.
Proof.
  intros equations n ty Hin. simpl. split.
  - apply in_flat_map. exists (n, ty). split; [ exact Hin | ].
    left. reflexivity.
  - intros equation Hequation. apply in_flat_map. exists (n, ty).
    split; [ exact Hin | ]. right. exact Hequation.
Qed.

Theorem equation_names_create_variant
  : equation_names_create_variant_statement.
Proof. intros is_unique immediates blocks extensions. reflexivity. Qed.

(* ======================================================================== *)
(* The variables of the aliases and of the names of equations are free      *)
(* ======================================================================== *)

(* The variables of some items are among some variables. *)
Section Covered.

Context {A : Type} (variables_of : A -> list variable).

Definition covered (items : list A) (xs : list variable) : Prop :=
  ∀ (a : A) (x : variable), a ∈ items -> x ∈ variables_of a -> x ∈ xs.

Lemma covered_nil (xs : list variable) : covered [] xs.
Proof. intros a x []. Qed.

Lemma covered_app (items items' : list A) (xs xs' : list variable) :
  covered items xs -> covered items' xs' ->
  covered (items ++ items') (xs ++ xs').
Proof.
  intros Hitems Hitems' a x Hin Hx. apply in_or_app.
  apply in_app_or in Hin as [Hin | Hin];
    [ left; exact (Hitems a x Hin Hx) | right; exact (Hitems' a x Hin Hx) ].
Qed.

Lemma covered_cons (a : A) (items : list A) (xs : list variable) :
  covered items xs -> covered (a :: items) (variables_of a ++ xs).
Proof.
  intros Hitems a' x [<- | Hin] Hx; apply in_or_app;
    [ left; exact Hx | right; exact (Hitems a' x Hin Hx) ].
Qed.

Lemma covered_right (items : list A) (xs xs' : list variable) :
  covered items xs' -> covered items (xs ++ xs').
Proof.
  intros Hitems a x Hin Hx. apply in_or_app. right.
  exact (Hitems a x Hin Hx).
Qed.

Lemma covered_flat_map {E : Type} (f : E -> list A)
    (g : E -> list variable) (l : list E) :
  (∀ e : E, e ∈ l -> covered (f e) (g e)) ->
  covered (flat_map f l) (flat_map g l).
Proof.
  intros Hl a x Hin Hx. apply in_flat_map in Hin as [e [He Hin]].
  apply in_flat_map. exists e. split; [ exact He | ].
  exact (Hl e He a x Hin Hx).
Qed.

End Covered.

(* The aliases. *)
Local Abbreviation alias_covered :=
  (covered (fun alias : simple * flambda_kind =>
              free_variables_of_simple (fst alias))).

Lemma aliases_of_relation_covered (x : option variable) :
  alias_covered (aliases_of_relation x) (free_variables_of_option x).
Proof.
  destruct x as [x | ]; [ | apply covered_nil ].
  intros alias y [<- | []] Hy. exact Hy.
Qed.

Lemma aliases_of_descr_covered {head : Type} (k : flambda_kind)
    (aliases_of_head : head -> list (simple * flambda_kind))
    (free_variables_of_head : head -> list variable) (ty : type_descr head) :
  (∀ h : head,
    ty = Ok (No_alias h) ->
    alias_covered (aliases_of_head h) (free_variables_of_head h)) ->
  alias_covered (aliases_of_descr k aliases_of_head ty)
    (free_variables_of_descr free_variables_of_head ty).
Proof.
  destruct ty as [ | [h | s] | ]; simpl; intros Hhead;
    [ apply covered_nil | exact (Hhead h eq_refl) | | apply covered_nil ].
  intros alias x [<- | []] Hx. exact Hx.
Qed.

Lemma aliases_of_block_case_covered (case : row_like_block_case) :
  P_block_case
    (fun t => alias_covered (aliases_with_kinds t) (free_variables t))
    (fun extension =>
       alias_covered (aliases_of_env_extension extension)
         (free_variables_of_env_extension extension))
    case ->
  alias_covered
    (aliases_of_block_case aliases_with_kinds aliases_of_env_extension case)
    (free_variables_of_block_case free_variables
       free_variables_of_env_extension case).
Proof.
  intros [Htypes Hextension]. apply covered_app; [ | exact Hextension ].
  apply covered_flat_map. exact Htypes.
Qed.

Lemma aliases_covered_all :
  (∀ t : flambda_type,
    alias_covered (aliases_with_kinds t) (free_variables t))
  ∧ (∀ head : head_of_kind_value,
      alias_covered (aliases_of_head_of_kind_value head)
        (free_variables_of_head_of_kind_value head))
  ∧ (∀ head : head_of_kind_value_non_null,
      alias_covered (aliases_of_head_of_kind_value_non_null head)
        (free_variables_of_head_of_kind_value_non_null head))
  ∧ (∀ blocks : row_like_for_blocks,
      alias_covered (aliases_of_row_like_for_blocks blocks)
        (free_variables_of_row_like_for_blocks blocks))
  ∧ (∀ closures : row_like_for_closures,
      alias_covered (aliases_of_row_like_for_closures closures)
        (free_variables_of_row_like_for_closures closures))
  ∧ (∀ entry : closures_entry,
      alias_covered (aliases_of_closures_entry entry)
        (free_variables_of_closures_entry entry))
  ∧ (∀ extension : env_extension,
      alias_covered (aliases_of_env_extension extension)
        (free_variables_of_env_extension extension)).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. simpl. apply aliases_of_descr_covered.
    intros h ->. exact Hhead.
  - intros ty. simpl. apply aliases_of_descr_covered.
    intros h _. apply covered_nil.
  - intros k ty. simpl. apply aliases_of_descr_covered.
    intros h _. apply covered_nil.
  - intros ty. simpl. apply aliases_of_descr_covered.
    intros h _. apply covered_nil.
  - intros ty. simpl. apply aliases_of_descr_covered.
    intros h _. apply covered_nil.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null. simpl. apply covered_app.
    + destruct non_null as [ | head | ];
        [ apply covered_nil | exact Hnon_null | apply covered_nil ].
    + destruct is_null as [ | x];
        [ apply covered_nil | apply aliases_of_relation_covered ].
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions.
    simpl. apply covered_app; [ apply aliases_of_relation_covered | ].
    apply covered_app.
    { destruct immediates as [ty | ];
        [ exact Himmediates | apply covered_nil ]. }
    apply covered_app; [ apply aliases_of_relation_covered | ].
    apply covered_app.
    { destruct blocks as [row | ]; [ exact Hblocks | apply covered_nil ]. }
    destruct extensions as [ | when_immediate when_block];
      [ apply covered_nil | ].
    exact (covered_app _ _ _ _ _ (proj1 Hextensions) (proj2 Hextensions)).
  - (* Mutable_block *)
    intros alloc_mode. apply covered_nil.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. exact Hcontents.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. exact Hclosures.
  - (* String *)
    intros strs. apply covered_nil.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    apply covered_app.
    + apply covered_flat_map. intros [tag [case | ]] Hin; simpl;
        [ | apply covered_nil ].
      exact (aliases_of_block_case_covered case (Hknown tag case Hin)).
    + destruct other_tags as [case | ]; [ | apply covered_nil ].
      exact (aliases_of_block_case_covered case Hother).
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl. apply covered_flat_map.
    intros [slot case] Hin. destruct (Hknown slot case Hin)
      as [Hentry Hextension].
    exact (covered_app _ _ _ _ _ Hentry Hextension).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. apply covered_app; apply covered_flat_map; intros [slot ty] Hin;
      [ exact (Hclosures slot ty Hin) | exact (Hslots slot ty Hin) ].
  - (* Env_extension *)
    intros equations Htypes. simpl. apply covered_flat_map.
    intros [n ty] Hin. simpl. apply covered_right. exact (Htypes n ty Hin).
Qed.

Theorem aliases_free_variables : aliases_free_variables_statement.
Proof.
  destruct aliases_covered_all as [Htype [_ [_ [_ [_ [_ Hextension]]]]]].
  split.
  - intros t s k x Hin Hx. exact (Htype t (s, k) x Hin Hx).
  - intros extension s k x Hin Hx. exact (Hextension extension (s, k) x Hin Hx).
Qed.

(* The names of equations. *)
Local Abbreviation equation_covered :=
  (covered (fun equation : name * flambda_kind =>
              free_variables_of_name (fst equation))).

Lemma equation_names_of_block_case_covered (case : row_like_block_case) :
  P_block_case
    (fun t =>
       equation_covered (equation_names_with_kinds t) (free_variables t))
    (fun extension =>
       equation_covered (equation_names_of_env_extension extension)
         (free_variables_of_env_extension extension))
    case ->
  equation_covered
    (equation_names_of_block_case equation_names_with_kinds
       equation_names_of_env_extension case)
    (free_variables_of_block_case free_variables
       free_variables_of_env_extension case).
Proof.
  intros [Htypes Hextension]. apply covered_app; [ | exact Hextension ].
  apply covered_flat_map. exact Htypes.
Qed.

Lemma equation_names_covered_all :
  (∀ t : flambda_type,
    equation_covered (equation_names_with_kinds t) (free_variables t))
  ∧ (∀ head : head_of_kind_value,
      equation_covered (equation_names_of_head_of_kind_value head)
        (free_variables_of_head_of_kind_value head))
  ∧ (∀ head : head_of_kind_value_non_null,
      equation_covered (equation_names_of_head_of_kind_value_non_null head)
        (free_variables_of_head_of_kind_value_non_null head))
  ∧ (∀ blocks : row_like_for_blocks,
      equation_covered (equation_names_of_row_like_for_blocks blocks)
        (free_variables_of_row_like_for_blocks blocks))
  ∧ (∀ closures : row_like_for_closures,
      equation_covered (equation_names_of_row_like_for_closures closures)
        (free_variables_of_row_like_for_closures closures))
  ∧ (∀ entry : closures_entry,
      equation_covered (equation_names_of_closures_entry entry)
        (free_variables_of_closures_entry entry))
  ∧ (∀ extension : env_extension,
      equation_covered (equation_names_of_env_extension extension)
        (free_variables_of_env_extension extension)).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. destruct ty as [ | [head | s] | ]; simpl;
      [ apply covered_nil | exact Hhead | apply covered_nil
      | apply covered_nil ].
  - intros ty. apply covered_nil.
  - intros k ty. apply covered_nil.
  - intros ty. apply covered_nil.
  - intros ty. apply covered_nil.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null.
    destruct non_null as [ | head | ]; simpl;
      [ apply covered_nil | | apply covered_nil ].
    intros equation x Hin Hx. apply in_or_app. left.
    exact (Hnon_null equation x Hin Hx).
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions.
    simpl. apply covered_right. apply covered_app.
    { destruct immediates as [ty | ];
        [ exact Himmediates | apply covered_nil ]. }
    apply covered_right. apply covered_app.
    { destruct blocks as [row | ]; [ exact Hblocks | apply covered_nil ]. }
    destruct extensions as [ | when_immediate when_block];
      [ apply covered_nil | ].
    exact (covered_app _ _ _ _ _ (proj1 Hextensions) (proj2 Hextensions)).
  - (* Mutable_block *)
    intros alloc_mode. apply covered_nil.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. exact Hcontents.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. exact Hclosures.
  - (* String *)
    intros strs. apply covered_nil.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    apply covered_app.
    + apply covered_flat_map. intros [tag [case | ]] Hin; simpl;
        [ | apply covered_nil ].
      exact (equation_names_of_block_case_covered case
               (Hknown tag case Hin)).
    + destruct other_tags as [case | ]; [ | apply covered_nil ].
      exact (equation_names_of_block_case_covered case Hother).
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl. apply covered_flat_map.
    intros [slot case] Hin. destruct (Hknown slot case Hin)
      as [Hentry Hextension].
    exact (covered_app _ _ _ _ _ Hentry Hextension).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. apply covered_app; apply covered_flat_map; intros [slot ty] Hin;
      [ exact (Hclosures slot ty Hin) | exact (Hslots slot ty Hin) ].
  - (* Env_extension *)
    intros equations Htypes. simpl. apply covered_flat_map.
    intros [n ty] Hin. simpl.
    exact (covered_cons _ (n, kind ty) _ _ (Htypes n ty Hin)).
Qed.

Theorem equation_names_free_variables
  : equation_names_free_variables_statement.
Proof.
  destruct equation_names_covered_all
    as [Htype [_ [_ [_ [_ [_ Hextension]]]]]].
  split.
  - intros t n k x Hin Hx. exact (Htype t (n, k) x Hin Hx).
  - intros extension n k x Hin Hx.
    exact (Hextension extension (n, k) x Hin Hx).
Qed.
